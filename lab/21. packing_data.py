from utils import *
import matplotlib.image as mpimg
import pickle

def make_dataset(src='data/**/*.jpeg', target='dataset.pkl'):
    cars, notcars, info = get_data_and_info(src)

    file_list = {}

    car = []
    not_car = []

    for file in cars:
        image = mpimg.imread(file)

        car.append(image)

    for file in notcars:
        image = mpimg.imread(file)
        not_car.append(image)

    file_list['car'] = car
    file_list['notcar'] = not_car

    pickle.dump(file_list, open(target, 'wb'))


if __name__ == "__main__":
    make_dataset('data/**/*.jpeg', 'simple_data.pkl')
    make_dataset('full_data/**/*.png', 'full_data.pkl')



