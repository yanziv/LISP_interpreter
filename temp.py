def flat(lst):


    if isinstance(lst, list):
        if len(lst) == 0:
            return []
        else:
            return flat(lst[0]) + flat(lst[1:])
    else:
        return [lst]
    

    if not isinstance(lst,list):
        return [lst]
    if len(lst) == 0:
        return []
    return flat(lst[0]) + flat(lst[1:])
    # output = []
    # for elem in lst:
    #     if not isinstance(elem,list):
    #         output.append(elem)
    #     else:
    #         output += flat(elem)
    # return output


arr = [1, [2,3], [[[[[4]]]]]]
print(flat(arr))