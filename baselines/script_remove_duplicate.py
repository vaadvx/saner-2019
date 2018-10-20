

with open("./test_vectors/cpp_vectors_bitbcnn_def_use.txt", "r") as f:
    data = f.readlines()

data_s = list(set(data))


for s in data_s:
	with open("./test_vectors/temp_2.txt", "a") as f:
		f.write(s)