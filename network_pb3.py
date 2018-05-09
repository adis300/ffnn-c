from protobuf3.message import Message
from protobuf3.fields import Int32Field, MessageField, DoubleField, EnumField
from enum import Enum


class Weight(Message):
    pass


class Bias(Message):
    pass


class Network(Message):

    class ActivationType(Enum):
        SIGMOID = 0
        LINEAR = 1
        RELU = 2
        THRESHOLD = 3
        SOFTMAX = 4

Weight.add_field('col', Int32Field(field_number=1, optional=True))
Weight.add_field('row', Int32Field(field_number=2, optional=True))
Weight.add_field('grid', DoubleField(field_number=3, repeated=True))
Bias.add_field('vector', DoubleField(field_number=1, repeated=True))
Network.add_field('layerSizes', Int32Field(field_number=1, repeated=True))
Network.add_field('activations', EnumField(field_number=2, repeated=True, enum_cls=Network.ActivationType))
Network.add_field('weights', MessageField(field_number=3, repeated=True, message_cls=Weight))
Network.add_field('biases', MessageField(field_number=4, repeated=True, message_cls=Bias))
