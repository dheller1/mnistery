from mnistery.load_mnist import load_data, reshape_data


def main():
    train_data, test_data = load_data()
    print(train_data[0].shape)
    train_data = reshape_data(train_data)
    print("bla")
    print(train_data[15][1])
    pass


if __name__ == '__main__':
    main()