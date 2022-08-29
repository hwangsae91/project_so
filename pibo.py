# numpy docstring 방식 주석

def pibo_recu(n:int) -> int:
    """
    recursive를 이용한 피보나치 수열

    Parameters
    ----------
    n : int
        구하려는 피보나치 수열의 순서

    Returns
    -------
    result: int
        n 번째 피보나치 수열의 값
    """

    result = n if n <= 1 else pibo_recu(n - 1) + pibo_recu(n - 2)

    return result

def pibo_dynamic(n:int) -> int:
    """
    dynamic algo를 이용한 피보나치 수열

    Parameters
    ----------
    n : int
        구하려는 피보나치 수열의 순서

    Returns
    -------
    result: int
        n 번째 피보나치 수열의 값
    """

    pibo_dict = {0: 0, 1: 1, 2: 1}

    result = None

    if n in pibo_dict:
        result = pibo_dict[n]
    else:
        result = pibo_recu(n - 1) + pibo_recu(n - 2)
    
    return result

print(pibo_dynamic(20))

