def generator2():
    for i in range(10):
        yield i

def generator3():
    for j in range(10, 20):
        yield j

def generator():
    for i in generator2():
        yield i
    for j in generator3():
        yield j


def generator_easy():
    yield from generator2()
    yield from generator3()

v = generator()
v1 = generator_easy()

for i in v:
    print(i)

print("----------")

for i in v1:
    print(i)