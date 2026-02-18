from parser_utils import parse_program
program = """
fractal: [256, 4], 
moe: [256, 16], 
gqa: [256, 8, 2], 
[256, 10]
"""
result = parse_program(program)
print(f"Result Length: {len(result) if result else 0}")
if result:
    for i, res in enumerate(result):
        print(f"Layer {i}: {res}")
else:
    print("Parsing Failed.")
