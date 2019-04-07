from fastai.vision import FloatList, ItemList, Iterator, Tensor


class ArraysItemList(FloatList):

    def __init__(self, items: Iterator, log: bool = False, **kwargs):
        if isinstance(items, ItemList):
            items = items.items
        super(FloatList, self).__init__(items, **kwargs)
    
    def get(self, i):
        float_get = super(FloatList, self).get(i).astype('float32')
        return Tensor(float_get)
