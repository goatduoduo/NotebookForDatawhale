def find_solution(ops, limits, target, solution):
    if target == 0:
        filter_best_solution(solution)
        return
    if not ops or not limits:
        return
    op = ops[0]
    limit = limits[0]
    # 开始迭代
    for i in range(1, limit):
        temp = target + op * i
        solution.append(op)
        find_solution(ops[1:], limits[1:], temp, solution)
    for i in range(1, limit):
        solution.pop()

    # 如果第一个就是0
    find_solution(ops[1:], limits[1:], target, solution)
    return


def filter_best_solution(temp_solution):
    global best_solution
    if not best_solution:
        best_solution = temp_solution[:]
        print("找到了可行解：", best_solution)
    if len(temp_solution) < len(best_solution):
        best_solution = temp_solution[:]
        print("找到了可行解：", best_solution)


ops = [28, 6, 12, 34, -11, -29, -37, -31]
limits = [3, 3, 3, 3, 3, 3, 3, 3]
target = -57
solution = []
best_solution = []

find_solution(ops, limits, target, solution)
print("找到了最优解：", best_solution)
for i in best_solution:
    target += i
print("验算：" + target.__str__())
