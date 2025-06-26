from colorama import Fore, Back, Style

class Result:
    def __init__(self, _params, _total_acc, _m_acc, _f_acc):
        self.total_acc = _total_acc
        self.m_acc = _m_acc
        self.f_acc = _f_acc

        self.params = _params

    def disparity(self):
        return abs(self.m_acc - self.f_acc)

    def performance(self):
        return self.total_acc - 0.5*self.disparity()

    def __cmp__(self, other):
        return self.performance() - other.performance()

    def __lt__(self, other):
        if self.performance() == other.performance():
            return self.f_acc > other.f_acc
        else:
            return self.performance() > other.performance()

    def __str__(self):
        text = "\n" + str(self.params) + "\n"
        text += "Acc \t Acc M \t\t Acc F \t Disparity\n"
        text += f"{Fore.GREEN}{self.total_acc:.4f} \t {Fore.BLUE}{self.m_acc:.4f} \t {Fore.RED}{self.f_acc:.4f}\t {Fore.CYAN}{self.disparity():.4f}"
        text += Fore.RESET
        return text
