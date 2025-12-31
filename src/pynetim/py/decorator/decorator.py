import time
from functools import wraps


class Timer:
    """
    计时器装饰器类，用于测量函数执行时间。

    该装饰器可以重复运行函数多次并计算平均执行时间。

    Attributes:
        repeat (int): 函数每次调用时实际运行的次数
        total_time (float): 累计总执行时间
        call_count (int): 函数被调用的次数
    """

    def __init__(self, repeat=1):
        """
        初始化计时器装饰器。

        Args:
            repeat (int, optional): 函数每次调用时实际运行的次数，默认为1
        """
        self.repeat = repeat
        self.total_time = 0
        self.call_count = 0

    def __call__(self, func):
        """
        使Timer实例可调用，作为装饰器使用。

        Args:
            func (Callable): 被装饰的函数

        Returns:
            Callable: 包装后的函数
        """
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            包装函数，执行实际的计时逻辑。

            Args:
                *args: 传递给被装饰函数的位置参数
                **kwargs: 传递给被装饰函数的关键字参数

            Returns:
                Any: 被装饰函数的返回值
            """
            result = None
            for _ in range(self.repeat):
                start_time = time.time()
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                self.total_time += elapsed
                self.call_count += 1
                # print(f"[Timer] {func.__name__} run {_ + 1}/{self.repeat}: {elapsed:.6f}s")
            print(f"运行时间: {average_time():.6f}s")
            return result

        def average_time():
            """
            计算函数的平均执行时间。

            Returns:
                float: 平均执行时间（秒）
            """
            if self.call_count == 0:
                return 0
            return self.total_time / self.call_count

        wrapper.average_time = average_time
        return wrapper
