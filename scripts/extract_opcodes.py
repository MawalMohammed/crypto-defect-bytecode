"""Disassemble fetched runtime bytecode into opcode token sequences (zero-dep).

- Strips the Solidity CBOR metadata trailer (last-2-bytes length suffix) before
  hashing, so functionally identical deployments hash identically for dedup.
- Emits opcode mnemonics only (PUSH immediates dropped, PUSHn kept as token).
  Exception: PUSH1..PUSH4 immediates that follow specific patterns are dropped
  too -- sequence models learn structure, not constants. The precompile-call
  signature (e.g. ecrecover = CALL preceded by small-constant address 0x1) stays
  visible via the opcode pattern itself.

Input : data/bytecode.jsonl
Output: data/opcodes.jsonl   {"address", "n_ops", "code_hash" (sha256 of
        metadata-stripped bytes), "ops" (space-joined mnemonics)}
        Addresses with empty code (0x) are skipped and listed in
        data/empty_code.txt
"""
import hashlib
import json
from pathlib import Path

DATA = Path(r"G:\Claude\new blockchain\crypto_defect_ml\data")

# EVM opcode table (mnemonics; unknown bytes -> INVALID_xx)
OPCODES = {
    0x00: "STOP", 0x01: "ADD", 0x02: "MUL", 0x03: "SUB", 0x04: "DIV", 0x05: "SDIV",
    0x06: "MOD", 0x07: "SMOD", 0x08: "ADDMOD", 0x09: "MULMOD", 0x0A: "EXP",
    0x0B: "SIGNEXTEND",
    0x10: "LT", 0x11: "GT", 0x12: "SLT", 0x13: "SGT", 0x14: "EQ", 0x15: "ISZERO",
    0x16: "AND", 0x17: "OR", 0x18: "XOR", 0x19: "NOT", 0x1A: "BYTE", 0x1B: "SHL",
    0x1C: "SHR", 0x1D: "SAR",
    0x20: "SHA3",
    0x30: "ADDRESS", 0x31: "BALANCE", 0x32: "ORIGIN", 0x33: "CALLER",
    0x34: "CALLVALUE", 0x35: "CALLDATALOAD", 0x36: "CALLDATASIZE",
    0x37: "CALLDATACOPY", 0x38: "CODESIZE", 0x39: "CODECOPY", 0x3A: "GASPRICE",
    0x3B: "EXTCODESIZE", 0x3C: "EXTCODECOPY", 0x3D: "RETURNDATASIZE",
    0x3E: "RETURNDATACOPY", 0x3F: "EXTCODEHASH",
    0x40: "BLOCKHASH", 0x41: "COINBASE", 0x42: "TIMESTAMP", 0x43: "NUMBER",
    0x44: "DIFFICULTY", 0x45: "GASLIMIT", 0x46: "CHAINID", 0x47: "SELFBALANCE",
    0x48: "BASEFEE", 0x49: "BLOBHASH", 0x4A: "BLOBBASEFEE",
    0x50: "POP", 0x51: "MLOAD", 0x52: "MSTORE", 0x53: "MSTORE8", 0x54: "SLOAD",
    0x55: "SSTORE", 0x56: "JUMP", 0x57: "JUMPI", 0x58: "PC", 0x59: "MSIZE",
    0x5A: "GAS", 0x5B: "JUMPDEST", 0x5C: "TLOAD", 0x5D: "TSTORE", 0x5E: "MCOPY",
    0x5F: "PUSH0",
    0xF0: "CREATE", 0xF1: "CALL", 0xF2: "CALLCODE", 0xF3: "RETURN",
    0xF4: "DELEGATECALL", 0xF5: "CREATE2", 0xFA: "STATICCALL", 0xFD: "REVERT",
    0xFE: "INVALID", 0xFF: "SELFDESTRUCT",
}
for i in range(32):
    OPCODES[0x60 + i] = f"PUSH{i + 1}"
for i in range(16):
    OPCODES[0x80 + i] = f"DUP{i + 1}"
    OPCODES[0x90 + i] = f"SWAP{i + 1}"
for i in range(5):
    OPCODES[0xA0 + i] = f"LOG{i}"


def strip_metadata(code: bytes) -> bytes:
    """Strip Solidity CBOR metadata trailer: last 2 bytes give trailer length."""
    if len(code) < 4:
        return code
    tlen = int.from_bytes(code[-2:], "big")
    # trailer = tlen bytes of CBOR + the 2 length bytes themselves
    if 0 < tlen <= 100 and tlen + 2 <= len(code):
        candidate = code[-(tlen + 2):-2]
        # sanity: solc metadata starts with CBOR map header 0xa1/0xa2/0xa3
        if candidate[:1] in (b"\xa1", b"\xa2", b"\xa3"):
            return code[: -(tlen + 2)]
    return code


def disassemble(code: bytes):
    ops = []
    i = 0
    n = len(code)
    while i < n:
        b = code[i]
        name = OPCODES.get(b)
        if name is None:
            ops.append(f"INVALID_{b:02x}")
            i += 1
            continue
        ops.append(name)
        if 0x60 <= b <= 0x7F:  # PUSH1..PUSH32: skip immediate bytes
            i += 1 + (b - 0x5F)
        else:
            i += 1
    return ops


def main():
    out_f = open(DATA / "opcodes.jsonl", "w", encoding="utf-8")
    empty = []
    n = 0
    for line in open(DATA / "bytecode.jsonl", encoding="utf-8"):
        d = json.loads(line)
        hexcode = d["code"]
        if hexcode in ("0x", "", None):
            empty.append(d["address"])
            continue
        code = bytes.fromhex(hexcode[2:])
        stripped = strip_metadata(code)
        ops = disassemble(stripped)
        rec = {
            "address": d["address"],
            "n_ops": len(ops),
            "code_hash": hashlib.sha256(stripped).hexdigest(),
            "ops": " ".join(ops),
        }
        out_f.write(json.dumps(rec) + "\n")
        n += 1
        if n % 10000 == 0:
            print(f"  {n:,} disassembled")
    out_f.close()
    (DATA / "empty_code.txt").write_text("\n".join(empty))
    print(f"done: {n:,} contracts disassembled, {len(empty)} empty (self-destructed)")


if __name__ == "__main__":
    main()
