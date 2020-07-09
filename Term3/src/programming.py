import numpy as np
import random


# constant

class Const:
    def __init__(self, data=0):
        self.data = data
        self.name = "Constant %d" % data
        self.index = -1
        self.is_register = False
    def get(self):
        return self.data
    def set(self, data):
        assert(False)


# register

class Register:
    def __init__(self, number, data=0):
        self.data = data
        self.name = "Register%d" % number
        self.index = number
        self.is_register = True
    def get(self):
        return self.data
    def set(self, data):
        self.data = data


def check_register(registers, lower_bound=-1048576, upper_bound=1048576):
    flag = False
    for register in registers:
        tmp = register.get()
        if ((tmp is None) or (tmp <= lower_bound) or (tmp >= upper_bound)):
            flag = True
            break
    return flag


# program counter

class PCounter:
    def __init__(self, data=0):
        self.data = data
        self.name = "PCounter"
        self.index = -1
        self.is_register = False
    def inc(self):
        self.data += 1
    def get(self):
        return self.data
    def set(self, data):
        self.data = data
    def reset(self):
        self.data = 0


# specification / ISA

class SPEC:
    NOP = lambda pc, r0, r1, r2: (pc.inc(), None)
    ADD = lambda pc, r0, r1, r2: (pc.inc(), r0.set(r1.get() + r2.get()))
    SUB = lambda pc, r0, r1, r2: (pc.inc(), r0.set(r1.get() - r2.get()))
    MUL = lambda pc, r0, r1, r2: (pc.inc(), r0.set(r1.get() * r2.get()))
    DIV = lambda pc, r0, r1, r2: (pc.inc(), r0.set(0 if (r2.get() == 0) else r1.get() // r2.get()))
    MOD = lambda pc, r0, r1, r2: (pc.inc(), r0.set(0 if (r2.get() == 0) else r1.get() % r2.get()))
    ADDI = lambda pc, r0, r1, c2: (pc.inc(), r0.set(r1.get() + c2.get()))
    BEQ = lambda pc, r0, r1, a2: (pc.set(pc.get() + a2.get()) if (r0.get() == r1.get()) else pc.inc(), None)
    BLT = lambda pc, r0, r1, a2: (pc.set(pc.get() + a2.get()) if (r0.get() < r1.get()) else pc.inc(), None)

    # NOTE: name, argc, argv, fun
    # argv=0: reg, argv=1: const, argv=2: pc(address)
    ISA = [
        ("NOP", 3, [0,0,0], NOP),
        ("ADD", 3, [0,0,0], ADD),
        ("SUB", 3, [0,0,0], SUB),
        # ("MUL", 3, [0,0,0], MUL),
        # ("DIV", 3, [0,0,0], DIV),
        # ("MOD", 3, [0,0,0], MOD),
        ("ADDI", 3, [0,0,1], ADDI),
        ("BEQ", 3, [0,0,2], BEQ),
        ("BLT", 3, [0,0,2], BLT),
    ]
    # TODO: reconstruct ISA......
    ISA_dictionary = {
        "NOP": NOP,
        "ADD": ADD,
        "SUB": SUB,
        # "MUL": MUL,
        # "DIV": DIV,
        # "MOD": MOD,
        "ADDI": ADDI,
        "BEQ": BEQ,
        "BLT": BLT,
    }
    REGS = [
      "Register0",
      "Register1",
      "Register2",
      "Register3",
    #   "Register4",
    #   "Register5",
    #   "Register6",
    #   "Register7",
    ]
    N_ISA = len(ISA)
    N_REG = len(REGS)


# simulator
# NOTE: carry out a sequence of instructions (following SPEC)

class Simulator:
    def __init__(self, specification):
        self.specification = specification
        self.pc = PCounter(0)
        self.regs = [Register(n) for n in range(specification.N_REG)]

    def __call__(self, program, limit=20):
        step = 0
        while (self.pc.get() < program.line):
            instr = program.code[self.pc.get()]
            opecode = instr.opecode.name
            opecode = self.specification.ISA_dictionary[opecode]
            operands = []
            for op in instr.operands:
                if op.is_register:
                    operands.append(self.regs[op.index])
                else:
                    operands.append(op)
            opecode(self.pc, *operands)
            if (self.pc.get() < 0):
                self.pc.set(0)
            step += 1
            # infinite loop
            if (step > limit or check_register(self.regs)):
                for r in self.regs:
                    r.set(None)
                    # print("INFINITE LOOP!")
                break
    
    def initialize(self, init_pc, init_regs):
        self.pc.set(init_pc)
        for idx in range(self.specification.N_REG):
            self.regs[idx].set(init_regs[idx])

    def show(self):
        print("  PC: ", self.pc, end="")
        print()
        print("REGS: ", [r.get() for r in self.regs], end="")

    def reset(self, pc=True, regs=True):
        if pc:
            self.pc.reset()
        if regs:
            for reg in self.regs:
                reg.set(0.0)
        
        

# opecode of a instruction
# FIXME: reconstruct not to use spec.
class Opecode:
    def __init__(self, name, argc, argv, fun):
        self.name = name
        self.argc = argc
        self.argv = argv
        self.fun = fun
        
    def __call__(self, *operands):
        self.fun(*operands)


# operand of a instruction (constant/register)

class Operand:
    def __init__(self, argv):
        self.argv = argv
        
    def get(self):
        return self.argv.get()
    
    def set(self, data):
        if (self.argv.is_register):
            self.argv.set(data)
        else:
            assert(False)


# instruction of a program

class Instruction:
    def __init__(self, opecode, *operands):
        self.opecode = opecode
        self.operands = list(operands)
        self.data = None
        
    def show(self):
        # DEBUG:
        if (self.opecode.name == "BEQ") or (self.opecode.name == "BLT"):
            print("%5s(%s)" % (self.opecode.name, ",".join([operand.name+"("+str(operand.data)+")" for operand in self.operands])), end="")
        else:
            print("%5s(%s)" % (self.opecode.name, ",".join([operand.name for operand in self.operands])), end="")
        
    def evaluate(self):
        self.opecode(*self.operands)
        self.data = self.operands[0].get()


# program (GENE)

class Program:

    # def reproduce(p):
    #     new_p = Programmer.copy(p)
    #     return new_p

    # def crossover(p, q):
    #     n_p = random.randrange(1, p.line-1)
    #     n_q = random.randrange(1, q.line-1)
    #     code_p_former, code_p_latter = p.cut(n_p)
    #     code_q_former, code_q_latter = q.cut(n_q)

    #     new_code_p = code_p_former + code_q_latter
    #     new_code_q = code_q_former + code_q_latter

    #     new_p = Programmer(p.spec, code=new_code_p)
    #     new_q = Programmer(q.spec, code=new_code_q)

    #     return new_p, new_q

    # def mutate(p):
    #     n = random.randrange(0, p.line)
    #     new_p = Programmer.copy(p)
    #     new_p.code[n] = Programmer.random_instruction()
    #     return new_p

    def __init__(self, code):
        self.code = code
        self.line = len(code)
    
    def show(self):
        for l, instruction in enumerate(self.code):
            print("%2d: " % l, end=""); instruction.show(); print()

    def cut(self, n):
        former = self.code[:n]
        latter = self.code[n:]
        return former, latter

    def get(self):
        return self.code

    def set(self, x):
        pass


# programmer (GENERATOR)

class Programmer:
    def __init__(self, specification, line):
        self.specification = specification
        self.line = line

    def __call__(self):
        program = self.random_program()
        return program

    # NOTE: shallow-copy (but entity of each instruction is the same as the original)
    def copy(self, p):
        code = p.code.copy()
        return Program(code=code)

    def reproduce(self, p):
        new_p = self.copy(p)
        return new_p

    def crossover(self, p, q):
        n_p = random.randrange(1, p.line-1)
        n_q = random.randrange(1, q.line-1)
        code_p_former, code_p_latter = p.cut(n_p)
        code_q_former, code_q_latter = q.cut(n_q)

        new_code_p = code_p_former + code_q_latter
        new_code_q = code_q_former + code_q_latter

        new_p = Program(code=new_code_p)
        new_q = Program(code=new_code_q)

        return new_p, new_q

    # NOTE: mutate: replace an instruction of (fixed-length) program
    def mutate(self, p):
        option = ["instruction", "program"]
        weight = [0.8, 0.2]
        # weight = [0.5, 0.5]
        choice = np.random.choice(option, p=weight)
        # TODO: implement parameter randomize
        if (choice == "parameter"):
            return p
        elif (choice == "instruction"):
            n = random.randrange(0, p.line)
            new_p = self.copy(p)
            new_p.code[n] = self.random_instruction()
            return new_p
        elif (choice == "program"):
            new_p = self.random_program()
            return new_p


    def random_instruction(self):
        idx = np.random.randint(0, self.specification.N_ISA)
        name, argc, argv, fun = self.specification.ISA[idx]
        opecode = Opecode(name, argc, argv, fun)
        operands = []
        for label in argv:
            if (label == 0):
                operands.append(Register(np.random.randint(0, self.specification.N_REG)))
            elif (label == 1):
                operands.append(Const(np.random.randint(-16,16)))
            elif (label == 2):
                operands.append(PCounter(np.random.randint(-self.line+1, self.line-1)))
            else:
                assert(False)
        instruction = Instruction(opecode, *operands)

        return instruction

    def random_program(self):
        instructions = [self.random_instruction() for _ in range(self.line)]
        program = Program(instructions)
        return program

    # def random_program(self, n=1, wrap=True):
    #     instructions = [self.random_instruction() for _ in range(n)]
    #     if wrap:
    #         program = Program(self, instructions)
    #         return program
    #     else:
    #         return instructions
