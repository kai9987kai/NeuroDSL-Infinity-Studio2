from parser_utils import parse_program
program = "fractal: [256], moe: [256, 8], [256, 1]"
result = parse_program(program)
print(f"Result: {result}")
