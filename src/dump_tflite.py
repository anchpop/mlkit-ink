#!/usr/bin/env python3
"""Dump a TFLite graph (ops, tensors, custom-op options) without needing to run it."""
import sys, collections
import tflite

def dump(path):
    buf = bytearray(open(path, "rb").read())
    model = tflite.Model.GetRootAsModel(buf, 0)
    print(f"version={model.Version()}  subgraphs={model.SubgraphsLength()}")
    opcodes = []
    for i in range(model.OperatorCodesLength()):
        oc = model.OperatorCodes(i)
        custom = oc.CustomCode()
        name = custom.decode() if custom else None
        if not custom:
            bi = oc.BuiltinCode() or oc.DeprecatedBuiltinCode()
            name = next((k for k, v in vars(tflite.BuiltinOperator).items() if isinstance(v,int) and v == bi), f"builtin#{bi}")
        opcodes.append(name)
    print("opcodes:", opcodes)

    for s in range(model.SubgraphsLength()):
        sg = model.Subgraphs(s)
        print(f"\n=== subgraph {s} ({sg.Name().decode() if sg.Name() else ''}) ===")
        print(" inputs:", list(sg.InputsAsNumpy()), " outputs:", list(sg.OutputsAsNumpy()))
        def tname(i):
            t = sg.Tensors(i)
            return f"{i}:{t.Name().decode()}{list(t.ShapeAsNumpy()) if t.ShapeLength() else []}"
        for o in range(sg.OperatorsLength()):
            op = sg.Operators(o)
            code = opcodes[op.OpcodeIndex()]
            print(f"\n op{o} {code}")
            print("   in :", [tname(i) if i >= 0 else "None" for i in op.InputsAsNumpy()])
            print("   out:", [tname(i) for i in op.OutputsAsNumpy()])
            n = op.CustomOptionsLength()
            if n:
                cust = bytes(op.CustomOptions(i) for i in range(n))
                import struct
                print(f"   custom_options ({n} bytes): {cust.hex(' ')}  f32[0]={struct.unpack('<f', cust[:4])[0]}  tail={list(cust[4:])}")
        print(f"\n --- tensors ({sg.TensorsLength()}) ---")
        for i in range(sg.TensorsLength()):
            t = sg.Tensors(i)
            tt = next((k for k, v in vars(tflite.TensorType).items() if isinstance(v,int) and v == t.Type()), str(t.Type()))
            b = model.Buffers(t.Buffer())
            nb = b.DataLength()
            print(f"  {i:3d} {t.Name().decode():55s} {str(list(t.ShapeAsNumpy()) if t.ShapeLength() else []):20s} {tt:8s} buf={nb}")

if __name__ == "__main__":
    dump(sys.argv[1])
