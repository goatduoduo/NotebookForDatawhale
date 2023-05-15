def find_solution(ops, limits, target, solution):
    if target == 0:
        return True
    if not ops or not limits:
        return False
    op = ops[0]
    limit = limits[0]
    for i in range(0, limit + 1):
        target += op
        solution.append(op)
        if find_solution(ops[1:], limits[1:], target, solution):
            return True
        target -= op
        solution.pop()
    return False

ops = [28,22,16,56,-19,-29,-37,-47]
limits = [11,11,6,8,9,8,9,6]
target = -28
solution = []

if find_solution(ops, limits, target, solution):
    print("找到了可行解：", solution)
    for i in solution:
        target += i
    print("验算："+target.__str__())
else:
    print("没有找到可行解")