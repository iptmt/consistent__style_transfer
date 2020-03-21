def myAtoi(str: str) -> int:
        # start 判断是否进入符号和数字判定，solid 判断是否进入0的判定
        start, solid, i_list, multip = False, False, [], 1
        for i in range(len(str)):
            c = ord(str[i])
            # Layer 0: 讨论非法字符，空格，合法字符的关系
            if c == 43 or c == 45 or 47 < c < 58:
                # Layer 1: 讨论正负号和数字的关系
                if 47 < c < 58:
                    # Layer 2: 讨论起始0的问题
                    if c == 48:
                        if solid:
                            i_list.append(c - 48)
                    else:
                        if not solid:
                            solid = True
                        i_list.append(c - 48)
                elif c == 45:
                    if not start:
                        multip = -1
                    else:
                        break
                elif c == 43:
                    if start:
                        break
                if not start:
                    start = True
            elif c <= 32:
                if start:
                    break
            else:
                break
        if not i_list:
            return 0     
        print(i_list)

        if multip == 1:
            max_int = 0x7fffffff
        else:
            max_int = 0x7fffffff + 1
        if len(i_list) > 10:
            return multip * max_int
        elif len(i_list) == 10:
            for i in range(10):
                base = 10 ** (9 - i)
                mx_cp = max_int // base
                max_int = max_int % base
                print(mx_cp, i_list[i])
                if mx_cp < i_list[i]:
                    if multip == 1:
                        return 0x7fffffff
                    else:
                        return -1 * (0x7fffffff + 1)
        num = 0
        for i in range(len(i_list)):
            base = 10 ** (len(i_list) - i - 1)
            num += i_list[i] * base
        return multip * num


print(myAtoi("1095502006p8"))