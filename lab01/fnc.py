def add_up(array):
    constant = 50000000
    while constant !=0:
        for i in range(len(array)):
            array[i] +=1
        constant -=1
    return array
