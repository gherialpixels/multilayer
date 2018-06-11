
def letters(filename):
    f = open(filename, 'r')
    length = 0
    count = [0.0 for i in range(26)]
    alph = [chr(97 + i) for i in range(26)]

    for line in f:
        for char in line:
            if not (char in alph):
                continue
            count[ord(char) - 97] += 1.0
            length += 1
    likelihood = [num / length for num in count]

    return likelihood

print letters("resources/nouns.txt")
