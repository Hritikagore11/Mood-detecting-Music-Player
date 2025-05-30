
import os
train_path = '/content/music/train'
test_path = '/content/music/test'

print("Train classes:", sorted(os.listdir(train_path)))
print("Test classes:", sorted(os.listdir(test_path)))
print("Train class count:", len(os.listdir(train_path)))
print("Test class count:", len(os.listdir(test_path)))
