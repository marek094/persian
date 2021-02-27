from persian.trainer import saved_torch_training
from schemas.schema_mnist_cnn import CnnMnistSchema

if __name__ == "__main__":
    parser = CnnMnistSchema.build_parser()
    args = parser.parse_args()

    model = CnnMnistSchema.from_args(args)
    saved_torch_training(model, verbose=True)