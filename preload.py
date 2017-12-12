from fileLoader import *
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.dirname(BASE_DIR))
data_dir = os.path.join(BASE_DIR, './scenes')
file = ['playground']

for f in file:
    TRAINING_FILE_LIST = os.path.join(data_dir, 'train_%s.txt' %f)
    TESTING_FILE_LIST = os.path.join(data_dir, 'test_%s.txt' %f)
    train_file_list = getDataFiles(TRAINING_FILE_LIST)
    num_train_file = len(train_file_list)
    test_file_list = getDataFiles(TESTING_FILE_LIST)
    num_test_file = len(test_file_list)
    for i in range(num_train_file):
        cur_train_filename = os.path.join(data_dir, '%s.obj' %train_file_list[i])
        print(i)
        obj = OBJ(cur_train_filename)
        points_seg = np.array(obj.vertices)
        savefilename = os.path.join(data_dir, 'test_%s' %train_file_list[i])
        np.save(savefilename, points_seg)

    for i in range(num_test_file):
        cur_train_filename = os.path.join(data_dir, '%s.obj' %test_file_list[i])
        print(i)
        obj = OBJ(cur_train_filename)
        points_seg = np.array(obj.vertices)
        savefilename = os.path.join(data_dir, 'test_%s' %test_file_list[i])
        np.save(savefilename, points_seg)

