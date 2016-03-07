
class MVNetworkDingo():
    """ DINGO network
    """
    def __init__(self, **kwargs):
        for attribute in ['buses', 'branches', 'transformers', 'sources']:
            setattr(self, attribute, kwargs.get(attribute, []))

        Entity.registry = self
        self.results = kwargs.get('results')
        self.time_idx = kwargs.get('time_idx')