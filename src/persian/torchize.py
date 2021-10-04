class SklearnTranform(object):

    def __init__(self, Cls, *args, **kwds):
        super().__init__()
        self.estimator = Cls(*args, **kwds)
        self.tnfm_before, self.tnfm_after = [], []

    def add(self, tnfm_before=None, tnfm_after=None):
        self.tnfm_before.append(tnfm_before)
        self.tnfm_after.append(tnfm_after)
        return self

    def __call__(self, input_diagram):
        if len(self.tnfm_before) > 0:
            for tn in self.tnfm_before:
                input_diagram = tn(input_diagram)
        result = self.estimator.fit_transform([input_diagram])[0]
        if len(self.tnfm_after) > 0:
            for tn in reversed(self.tnfm_after):
                result = tn(result)
        return result
