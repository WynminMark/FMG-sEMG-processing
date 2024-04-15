import math


def wjh_calculete_R(a, b, c):
    """
    通过任意三角形计算外接圆半径
    a,b,c: 三角形的三个边长
    """
    p = (a+b+c)/2
    R = a*b*c/(4*math.sqrt(p*(p-a)*(p-b)*(p-c)))
    return R

def zpk_calculate_R(h, c):
    """
    通过等腰三角形计算外接圆半径
    a: 等腰三角形的腰长
    h: 等腰三角形底边对应的高，如果可以直接准确测量 h，可以将输入改成h，并注释掉 h 计算语句
    c: 等腰三角形底边的长
    """
    # h = math.sqrt(math.pow(a,2) - math.pow(c/2, 2))
    R = math.pow(c, 2)/(2*h)
    return R

if __name__ == '__main__':
    print(wjh_calculete_R(4.99, 5, 9.8))
    print(zpk_calculate_R(5, 9.8))
    pass