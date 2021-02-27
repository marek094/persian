import unittest
from persian.trainer import validated_training
from persian.schema_protocol import Schema


class TestSchema(Schema):

    @staticmethod
    def list_hparams():
        return [
            dict(name='epochs', type=int, default=10),
            dict(name='decrease_per_epoch', type=float, default=0.096),
            dict(name='starting_error', type=float, default=1),
        ]

    def __init__(self, flags={}):
        super().__init__(flags)

    def prepare_dataset(self, set):
        pass

    def prepare_model(self):
        kInit = [1234, 257]
        self.weights = kInit
        self.model = lambda x: x * kInit[0] % kInit[1]

    def _cmp_metrics(self):
        return {
            'TRAIN':
                dict(loss=self.loss, acc=max(0, 1 - self.loss)),
            'VALID':
                dict(loss=self.loss + 0.05, acc=max(0, 1 - self.loss - 0.05)),
        }

    def prepare_criterium(self):
        self.loss = self.flags['starting_error']
        self.metrics = self._cmp_metrics()

    def epoch_range(self):
        return range(self.flags['epochs'])

    def run_batches(self, set_name):
        if set_name == "TRAIN":
            self.loss -= self.flags['decrease_per_epoch']
            self.metrics = self._cmp_metrics()

    def run_inference(self, input):
        output = self.model(input)
        return (output, self.metrics['VALID'])

    def pack_model_params(self):
        return self.weights

    def load_model_params(self, weights):
        self.weights = weights


class TestSchemaTraining(unittest.TestCase):

    def test_training(self):
        inst = TestSchema(dict(epochs=11, decrease_per_epoch=0.08))
        hstr = inst.as_hstr()

        self.assertEqual(hstr.split('_')[0], 'TestSchema')

        inst2 = TestSchema.from_hstr(hstr)

        self.assertAlmostEqual(inst2.flags['decrease_per_epoch'], 0.08)
        self.assertAlmostEqual(inst2.flags['starting_error'], 1.0)
        self.assertEqual(inst2.flags['epochs'], 11)

        tinst = validated_training(inst)

        self.assertAlmostEqual(tinst.metrics['TRAIN']['loss'], 1.0 - 11 * 0.08)
        self.assertAlmostEqual(tinst.metrics['VALID']['loss'],
                               1.0 - 11 * 0.08 + 0.05)
        self.assertAlmostEqual(tinst.metrics['VALID']['acc'], 11 * 0.08 - 0.05)


if __name__ == "__main__":
    unittest.main()
