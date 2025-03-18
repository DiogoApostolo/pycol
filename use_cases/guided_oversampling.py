from complexity import Complexity

comp = Complexity(file_name="dataset/winequality_red_4.arff")

B,S,R,O,C = comp.borderline(return_all=True,imb=True)

print(S)
print(B)
print(R)
print(O)