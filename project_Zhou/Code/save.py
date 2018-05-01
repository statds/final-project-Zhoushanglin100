import matplotlib.pyplot as plt
import numpy as np

with open("result", "r") as f:
	result = f.read().splitlines()
f.close()

result = filter(lambda e: "val_acc" in e, result)
result = map(lambda e: e.split("val_acc: ")[1], result)
result = map(float, result)

print result[900:]
print '------------------------'
print max(result)
print '------------------------'
print np.mean(result[900:])

plt.plot(result)
plt.ylabel('val_acc')
plt.xlabel('epoch')
#plt.show()
plt.savefig('val_acc_1')