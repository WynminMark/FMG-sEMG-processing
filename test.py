class A:
    def hi(self):
        self.a = 1

class B(A):
    def hi(self):
        super().hi()         # 通过super调用父类A的hi()
        self.a = 2
        print(self.a)
        
b = B()
b.hi()    # 这里调用的就是B里面的hi()
