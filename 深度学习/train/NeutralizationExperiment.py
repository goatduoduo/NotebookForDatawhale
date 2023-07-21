def find_solution(target, red_numbers, blue_numbers, red_limits, blue_limits, solution):
    # 检查是否已经达到目标
    if target == 0:
        return True

    # 尝试执行红色操作
    for i in range(len(red_numbers)):
        if red_limits[i] > 0:
            red_limits[i] -= 1
            solution.append("执行红色 {} 1次".format(red_numbers[i]))
            if find_solution(target + red_numbers[i], red_numbers, blue_numbers, red_limits, blue_limits, solution):
                return True
            # 回溯
            red_limits[i] += 1
            solution.pop()

    # 尝试执行蓝色操作
    for i in range(len(blue_numbers)):
        if blue_limits[i] > 0:
            blue_limits[i] -= 1
            solution.append("执行蓝色 {} 1次".format(blue_numbers[i]))
            if find_solution(target - blue_numbers[i], red_numbers, blue_numbers, red_limits, blue_limits, solution):
                return True
            # 回溯
            blue_limits[i] += 1
            solution.pop()

    return False


def main():
    target_number = 20
    red_numbers = [28, 22, 16, 56]
    blue_numbers = [19, 29, 37, 47]
    red_limits = [11, 11, 6, 8]
    blue_limits = [9, 8, 9, 6]
    solution = []

    if find_solution(target_number, red_numbers, blue_numbers, red_limits, blue_limits, solution):
        for step in solution:
            print(step)
    else:
        print("无解")


if __name__ == "__main__":
    main()
