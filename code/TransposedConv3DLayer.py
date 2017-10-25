
import theano
from theano import tensor as T

from theano.tensor import nnet 
from theano.tensor.nnet import abstract_conv
from theano.tensor.nnet.abstract_conv import AbstractConv3d_gradInputs
import theano.tensor.nnet.abstract_conv.BaseAbstractConv

import time
import lasagne
from lasagne.utils import as_tuple
from lasagne.layers.conv import BaseConvLayer, conv_output_length, conv_input_length


from lasagne.layers import InputLayer, NINLayer, flatten, reshape, Upscale3DLayer
from lasagne.nonlinearities import rectify as relu
from lasagne.nonlinearities import LeakyRectify as lrelu
from lasagne.nonlinearities import sigmoid
from lasagne.layers import get_all_params, get_all_layers
from lasagne.objectives import binary_crossentropy as bce
from lasagne.objectives import squared_error
from lasagne.updates import adam

from lasagne.layers import conv,base

from lasagne.theano_extensions import conv
from lasagne.layers import __init__


class TransposedConv3DLayer(BaseConvLayer):  # pragma: no cover
    """
    lasagne.layers.TransposedConv3DLayer(incoming, num_filters, filter_size,
    stride=(1, 1, 1), crop=0, untie_biases=False,
    W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
    nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False, **kwargs)
    3D transposed convolution layer
    Performs the backward pass of a 3D convolution (also called transposed
    convolution, fractionally-strided convolution or deconvolution in the
    literature) on its input and optionally adds a bias and applies an
    elementwise nonlinearity.
    Parameters
    ----------
    incoming : a :class:`Layer` instance or a tuple
        The layer feeding into this layer, or the expected input shape. The
        output of this layer should be a 5D tensor, with shape
        ``(batch_size, num_input_channels, input_depth, input_rows,
        input_columns)``.
    num_filters : int
        The number of learnable convolutional filters this layer has.
    filter_size : int or iterable of int
        An integer or a 3-element tuple specifying the size of the filters.
    stride : int or iterable of int
        An integer or a 3-element tuple specifying the stride of the
        transposed convolution operation. For the transposed convolution, this
        gives the dilation factor for the input -- increasing it increases the
        output size.
    crop : int, iterable of int, 'full', 'same' or 'valid' (default: 0)
        By default, the transposed convolution is computed where the input and
        the filter overlap by at least one position (a full convolution). When
        ``stride=1``, this yields an output that is larger than the input by
        ``filter_size - 1``. It can be thought of as a valid convolution padded
        with zeros. The `crop` argument allows you to decrease the amount of
        this zero-padding, reducing the output size. It is the counterpart to
        the `pad` argument in a non-transposed convolution.
        A single integer results in symmetric cropping of the given size on all
        borders, a tuple of three integers allows different symmetric cropping
        per dimension.
        ``'full'`` disables zero-padding. It is is equivalent to computing the
        convolution wherever the input and the filter fully overlap.
        ``'same'`` pads with half the filter size (rounded down) on both sides.
        When ``stride=1`` this results in an output size equal to the input
        size. Even filter size is not supported.
        ``'valid'`` is an alias for ``0`` (no cropping / a full convolution).
        Note that ``'full'`` and ``'same'`` can be faster than equivalent
        integer values due to optimizations by Theano.
    untie_biases : bool (default: False)
        If ``False``, the layer will have a bias parameter for each channel,
        which is shared across all positions in this channel. As a result, the
        `b` attribute will be a vector (1D).
        If True, the layer will have separate bias parameters for each
        position in each channel. As a result, the `b` attribute will be a
        3D tensor.
    W : Theano shared variable, expression, numpy array or callable
        Initial value, expression or initializer for the weights.
        These should be a 5D tensor with shape
        ``(num_input_channels, num_filters, filter_rows, filter_columns)``.
        Note that the first two dimensions are swapped compared to a
        non-transposed convolution.
        See :func:`lasagne.utils.create_param` for more information.
    b : Theano shared variable, expression, numpy array, callable or ``None``
        Initial value, expression or initializer for the biases. If set to
        ``None``, the layer will have no biases. Otherwise, biases should be
        a 1D array with shape ``(num_filters,)`` if `untied_biases` is set to
        ``False``. If it is set to ``True``, its shape should be
        ``(num_filters, output_rows, output_columns)`` instead.
        See :func:`lasagne.utils.create_param` for more information.
    nonlinearity : callable or None
        The nonlinearity that is applied to the layer activations. If None
        is provided, the layer will be linear.
    flip_filters : bool (default: False)
        Whether to flip the filters before sliding them over the input,
        performing a convolution, or not to flip them and perform a
        correlation (this is the default). Note that this flag is inverted
        compared to a non-transposed convolution.
    output_size : int or iterable of int or symbolic tuple of ints
        The output size of the transposed convolution. Allows to specify
        which of the possible output shapes to return when stride > 1.
        If not specified, the smallest shape will be returned.
    **kwargs
        Any additional keyword arguments are passed to the `Layer` superclass.
    Attributes
    ----------
    W : Theano shared variable or expression
        Variable or expression representing the filter weights.
    b : Theano shared variable or expression
        Variable or expression representing the biases.
    Notes
    -----
    The transposed convolution is implemented as the backward pass of a
    corresponding non-transposed convolution. It can be thought of as dilating
    the input (by adding ``stride - 1`` zeros between adjacent input elements),
    padding it with ``filter_size - 1 - crop`` zeros, and cross-correlating it
    with the filters. See [1]_ for more background.
    Examples
    --------
    To transpose an existing convolution, with tied filter weights:
    >>> from lasagne.layers import Conv3DLayer, TransposedConv3DLayer
    >>> conv = Conv3DLayer((None, 1, 32, 32, 32), 16, 3, stride=2, pad=2)conv
    >>> deconv = TransposedConv3DLayer(conv, conv.input_shape[1],
    ...         conv.filter_size, stride=conv.stride, crop=conv.pad,
    ...         W=conv.W, flip_filters=not conv.flip_filters)
    References
    ----------
    .. [1] Vincent Dumoulin, Francesco Visin (2016):
           A guide to convolution arithmetic for deep learning. arXiv.
           http://arxiv.org/abs/1603.07285,
           https://github.com/vdumoulin/conv_arithmetic
    """
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1, 1), crop=0, untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False,output_size=None, **kwargs):
        # output_size must be set before calling the super constructor
        if (not isinstance(output_size, T.Variable) and output_size is not None):
            output_size = as_tuple(output_size, 3, int)
        self.output_size = output_size
        BaseConvLayer.__init__(self, incoming, num_filters, filter_size, stride, crop, untie_biases, W, b,nonlinearity, flip_filters, n=3, **kwargs)
        # rename self.pad to self.crop:
        #if crop is None:
        self.crop = self.pad
        del self.pad
            

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        # first two sizes are swapped compared to a forward convolution
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        if self.output_size is not None:
            size = self.output_size
            if isinstance(self.output_size, T.Variable):
                size = (None, None, None)

            return input_shape[0], self.num_filters, size[0], size[1], size[2]

        # If self.output_size is not specified, return the smallest shape
        # when called from the constructor, self.crop is still called self.pad:
        crop = getattr(self, 'crop', getattr(self, 'pad', None))
        crop = crop if isinstance(crop, tuple) else (crop,) * self.n
        batchsize = input_shape[0]
        return ((batchsize, self.num_filters) + tuple(conv_input_length(input, filter, stride, p) for input, filter, stride, p in zip(input_shape[2:], self.filter_size, self.stride, crop)))

    def convolve(self, input, **kwargs):
        border_mode = 'half' if self.crop == 'same' else self.crop
        op = T.nnet.abstract_conv.AbstractConv3d_gradInputs(imshp=self.output_shape,kshp=self.get_W_shape(),subsample=self.stride, border_mode=border_mode,filter_flip=not self.flip_filters)
        output_size = self.output_shape[2:]
        if isinstance(self.output_size, T.Variable):
            output_size = self.output_size
        elif any(s is None for s in output_size):
            output_size = self.get_output_shape_for(input.shape)[2:]
        conved = op(self.W, input, output_size)
        return conved

Deconv3DLayer = TransposedConv3DLayer