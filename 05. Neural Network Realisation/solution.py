from homework_5.interface import *


# ================================= 1.4.1 SGD ================================
class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            return parameter - self.lr * parameter_grad
            # your code here /\

        return updater


# ============================= 1.4.2 SGDMomentum ============================
class SGDMomentum(Optimizer):
    def __init__(self, lr, momentum=0.0):
        self.lr = lr
        self.momentum = momentum

    def get_parameter_updater(self, parameter_shape):
        """
        :param parameter_shape: tuple, the shape of the associated parameter

        :return: the updater function for that parameter
        """

        def updater(parameter, parameter_grad):
            """
            :param parameter: np.array, current parameter values
            :param parameter_grad: np.array, current gradient, dLoss/dParam

            :return: np.array, new parameter values
            """
            # your code here \/
            updater.inertia = self.momentum * updater.inertia + self.lr * parameter_grad
            return parameter - updater.inertia
            # your code here /\

        updater.inertia = np.zeros(parameter_shape)
        return updater


# ================================ 2.1.1 ReLU ================================
class ReLU(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        return np.maximum(0, inputs)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        mask = (self.forward_inputs >= 0).astype(float)
        return grad_outputs * mask
        # your code here /\


# =============================== 2.1.2 Softmax ==============================
class Softmax(Layer):
    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, d)), output values

            n - batch size
            d - number of units
        """
        # your code here \/
        shifted_inputs = inputs - np.max(inputs, axis=1, keepdims=True)
        exp_inputs = np.exp(shifted_inputs)
        return exp_inputs / np.sum(exp_inputs, axis=1, keepdims=True)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of units
        """
        # your code here \/
        dot_product = np.sum(self.forward_outputs * grad_outputs, axis=1, keepdims=True)
        return self.forward_outputs * (grad_outputs - dot_product)
        # your code here /\


# ================================ 2.1.3 Dense ===============================
class Dense(Layer):
    def __init__(self, units, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_units = units

        self.weights, self.weights_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        (input_units,) = self.input_shape
        output_units = self.output_units

        # Register weights and biases as trainable parameters
        # Note, that the parameters and gradients *must* be stored in
        # self.<p> and self.<p>_grad, where <p> is the name specified in
        # self.add_parameter

        self.weights, self.weights_grad = self.add_parameter(
            name="weights",
            shape=(output_units, input_units),
            initializer=he_initializer(input_units),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_units,),
            initializer=np.zeros,
        )

        self.output_shape = (output_units,)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d)), input values

        :return: np.array((n, c)), output values

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        return np.dot(inputs, self.weights.T) + self.biases
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c)), dLoss/dOutputs

        :return: np.array((n, d)), dLoss/dInputs

            n - batch size
            d - number of input units
            c - number of output units
        """
        # your code here \/
        self.weights_grad[:] = np.dot(grad_outputs.T, self.forward_inputs)
        self.biases_grad[:] = np.sum(grad_outputs, axis=0)
        return np.dot(grad_outputs, self.weights)
        # your code here /\


# ============================ 2.2.1 Crossentropy ============================
class CategoricalCrossentropy(Loss):
    def value_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((1,)), mean Loss scalar for batch

            n - batch size
            d - number of units
        """
        # your code here \/
        y_pred = np.clip(y_pred, eps, 1 - eps)
        losse = -np.sum(y_gt * np.log(y_pred), axis=1)
        return np.array([np.mean(losse)])
        # your code here /\

    def gradient_impl(self, y_gt, y_pred):
        """
        :param y_gt: np.array((n, d)), ground truth (correct) labels
        :param y_pred: np.array((n, d)), estimated target values

        :return: np.array((n, d)), dLoss/dY_pred

            n - batch size
            d - number of units
        """
        # your code here \/
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return -y_gt / (y_pred * y_gt.shape[0])
        # your code here /\


# ======================== 2.3 Train and Test on MNIST =======================
def train_mnist_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = Model(
        loss=CategoricalCrossentropy(),
        optimizer=SGDMomentum(lr=0.01, momentum=0.9)
    )

    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)
    model.add(Dense(256, input_shape=(784,)))
    model.add(ReLU())

    model.add(Dense(64))
    model.add(ReLU())

    model.add(Dense(10))
    model.add(Softmax())

    print(model)

    # 3) Train and validate the model using the provided data
    model.fit(
        x_train=x_train,
        y_train=y_train,
        batch_size=128,
        epochs=20,
        shuffle=True,
        verbose=True,
        x_valid=x_valid,
        y_valid=y_valid
    )

    # your code here /\
    return model


# ============================== 3.3.2 convolve ==============================
def convolve(inputs, kernels, padding=0):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # !!! Don't change this function, it's here for your reference only !!!
    assert isinstance(padding, int) and padding >= 0
    assert inputs.ndim == 4 and kernels.ndim == 4
    assert inputs.shape[1] == kernels.shape[1]

    if os.environ.get("USE_FAST_CONVOLVE", False):
        return convolve_pytorch(inputs, kernels, padding)
    else:
        return convolve_numpy(inputs, kernels, padding)


def convolve_numpy(inputs, kernels, padding):
    """
    :param inputs: np.array((n, d, ih, iw)), input values
    :param kernels: np.array((c, d, kh, kw)), convolution kernels
    :param padding: int >= 0, the size of padding, 0 means 'valid'

    :return: np.array((n, c, oh, ow)), output values

        n - batch size
        d - number of input channels
        c - number of output channels
        (ih, iw) - input image shape
        (oh, ow) - output image shape
    """
    # your code here \/
    n, d, ih, iw = inputs.shape
    c, d_k, kh, kw = kernels.shape
    kernels_flipped = kernels[:, :, ::-1, ::-1]

    if padding > 0:
        inputs_padded = np.pad(
            inputs, 
            pad_width=((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
            constant_values=0
        )
    else:
        inputs_padded = inputs

    oh = ih + 2 * padding - kh + 1
    ow = iw + 2 * padding - kw + 1 

    output = np.zeros((n, c, oh, ow))
    for i in range(oh):
        for j in range(ow):
            window = inputs_padded[:, :, i:i+kh, j:j+kw]
            output[:, :, i, j] = np.sum(
                window[:, np.newaxis, :, :, :] * kernels_flipped[np.newaxis, :, :, :, :],
                axis=(2, 3, 4))

    return output        
    # your code here /\


# =============================== 4.1.1 Conv2D ===============================
class Conv2D(Layer):
    def __init__(self, output_channels, kernel_size=3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert kernel_size % 2, "Kernel size should be odd"

        self.output_channels = output_channels
        self.kernel_size = kernel_size

        self.kernels, self.kernels_grad = None, None
        self.biases, self.biases_grad = None, None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        output_channels = self.output_channels
        kernel_size = self.kernel_size

        self.kernels, self.kernels_grad = self.add_parameter(
            name="kernels",
            shape=(output_channels, input_channels, kernel_size, kernel_size),
            initializer=he_initializer(input_h * input_w * input_channels),
        )

        self.biases, self.biases_grad = self.add_parameter(
            name="biases",
            shape=(output_channels,),
            initializer=np.zeros,
        )

        self.output_shape = (output_channels,) + self.input_shape[1:]

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, c, h, w)), output values

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        padding = (self.kernel_size - 1) // 2
        return convolve(inputs, self.kernels, padding=padding) + self.biases[None, :, None, None]
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, c, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of input channels
            c - number of output channels
            (h, w) - image shape
        """
        # your code here \/
        padding = (self.kernel_size - 1) // 2
        
        self.biases_grad += np.sum(grad_outputs, axis=(0, 2, 3))
        inputs_transposed = self.forward_inputs.transpose(1, 0, 2, 3)
        grad_outputs_transposed = grad_outputs.transpose(1, 0, 2, 3)[:, :, ::-1, ::-1]
        
        kernels_grad = convolve(inputs_transposed, grad_outputs_transposed, padding=padding).transpose(1, 0, 2, 3)
        self.kernels_grad += kernels_grad[:, :, ::-1, ::-1]
        kernels_flipped = self.kernels[:, :, ::-1, ::-1].transpose(1, 0, 2, 3)
        
        return convolve(grad_outputs, kernels_flipped, padding=padding)
        # your code here /\

# ============================== 4.1.2 Pooling2D =============================
class Pooling2D(Layer):
    def __init__(self, pool_size=2, pool_mode="max", *args, **kwargs):
        super().__init__(*args, **kwargs)
        assert pool_mode in {"avg", "max"}

        self.pool_size = pool_size
        self.pool_mode = pool_mode
        self.forward_idxs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        channels, input_h, input_w = self.input_shape
        output_h, rem_h = divmod(input_h, self.pool_size)
        output_w, rem_w = divmod(input_w, self.pool_size)
        assert not rem_h, "Input height should be divisible by the pool size"
        assert not rem_w, "Input width should be divisible by the pool size"

        self.output_shape = (channels, output_h, output_w)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, ih, iw)), input values

        :return: np.array((n, d, oh, ow)), output values

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n, d, ih, iw = inputs.shape
        oh, ow = ih // self.pool_size, iw // self.pool_size

        inputs_reshaped = inputs.reshape(n, d, oh, self.pool_size, ow, self.pool_size)
        if self.pool_mode == "max":
            output = inputs_reshaped.max(axis=3).max(axis=4)
            max_indx = inputs_reshaped.argmax(axis=3).argmax(axis=4)
            self.forward_idxs = max_indx
        else:
            output = inputs_reshaped.mean(axis=3).mean(axis=4)    

        return output        
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, oh, ow)), dLoss/dOutputs

        :return: np.array((n, d, ih, iw)), dLoss/dInputs

            n - batch size
            d - number of channels
            (ih, iw) - input image shape
            (oh, ow) - output image shape
        """
        # your code here \/
        n, d, oh, ow = grad_outputs.shape
        ih, iw = oh * self.pool_size, ow * self.pool_size

        grad_inputs = np.zeros((n, d, ih, iw))
        if self.pool_mode == "max":
            for i in range(oh):
                for j in range(ow):
                    h_start, h_end = i * self.pool_size, (i + 1) * self.pool_size
                    w_start, w_end = j * self.pool_size, (j + 1) * self.pool_size
                    
                    window = self.forward_inputs[:, :, h_start:h_end, w_start:w_end].reshape(n, d, self.pool_size * self.pool_size)
                    max_positions = window.argmax(axis=2)
                    
                    for batch in range(n):
                        for chnl in range(d):
                            max_pos = max_positions[batch, chnl]
                            h_pos = h_start + max_pos // self.pool_size
                            w_pos = w_start + max_pos % self.pool_size
                            grad_inputs[batch, chnl, h_pos, w_pos] += grad_outputs[batch, chnl, i, j]
        else:
             for i in range(oh):
                for j in range(ow):
                    h_start, h_end = i * self.pool_size, (i + 1) * self.pool_size
                    w_start, w_end = j * self.pool_size, (j + 1) * self.pool_size
                    grad_inputs[:, :, h_start:h_end, w_start:w_end] += grad_outputs[:, :, i:i+1, j:j+1] / (self.pool_size * self.pool_size)
        
        return grad_inputs                    
        # your code here /\


# ============================== 4.1.3 BatchNorm =============================
class BatchNorm(Layer):
    def __init__(self, momentum=0.9, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.momentum = momentum

        self.running_mean = None
        self.running_var = None

        self.beta, self.beta_grad = None, None
        self.gamma, self.gamma_grad = None, None

        self.forward_inverse_std = None
        self.forward_centered_inputs = None
        self.forward_normalized_inputs = None

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        input_channels, input_h, input_w = self.input_shape
        self.running_mean = np.zeros((input_channels,))
        self.running_var = np.ones((input_channels,))

        self.beta, self.beta_grad = self.add_parameter(
            name="beta",
            shape=(input_channels,),
            initializer=np.zeros,
        )

        self.gamma, self.gamma_grad = self.add_parameter(
            name="gamma",
            shape=(input_channels,),
            initializer=np.ones,
        )

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, d, h, w)), output values

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        d = inputs.shape[1]
        if self.is_training:
            mu = np.mean(inputs, axis=(0, 2, 3))
            var = np.var(inputs, axis=(0, 2, 3))
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mu
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var

            self.forward_centered_inputs = inputs - mu.reshape(1, d, 1, 1)
            self.forward_inverse_std = 1.0 / np.sqrt(var + eps)
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std.reshape(1, d, 1, 1)

        else:
            self.forward_centered_inputs = inputs - self.running_mean.reshape(1, d, 1, 1)
            self.forward_inverse_std = 1.0 / np.sqrt(self.running_var + eps)
            self.forward_normalized_inputs = self.forward_centered_inputs * self.forward_inverse_std.reshape(1, d, 1, 1)

        output = self.gamma.reshape(1, d, 1, 1) * self.forward_normalized_inputs + self.beta.reshape(1, d, 1, 1)    
        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, d, h, w)), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of channels
            (h, w) - image shape
        """
        # your code here \/
        n, d, h, w = grad_outputs.shape
        m = n * h * w

        self.gamma_grad += np.sum(grad_outputs * self.forward_normalized_inputs, axis=(0, 2, 3))
        self.beta_grad += np.sum(grad_outputs, axis=(0, 2, 3))

        norm_grad = grad_outputs * self.gamma.reshape(1, d, 1, 1)
        var_grad = np.sum(norm_grad * self.forward_centered_inputs * -0.5 * (self.forward_inverse_std ** 3).reshape(1, d, 1, 1), 
                      axis=(0, 2, 3))
        mean_grad = np.sum(norm_grad * -self.forward_inverse_std.reshape(1, d, 1, 1), axis=(0, 2, 3))
        mean_grad += var_grad * np.mean(-2.0 * self.forward_centered_inputs, axis=(0, 2, 3))
        
        grad_inputs = (norm_grad * self.forward_inverse_std.reshape(1, d, 1, 1) +
                      var_grad.reshape(1, d, 1, 1) * 2.0 * self.forward_centered_inputs / m +
                      mean_grad.reshape(1, d, 1, 1) / m)
        
        return grad_inputs
        # your code here /\


# =============================== 4.1.4 Flatten ==============================
class Flatten(Layer):
    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

        self.output_shape = (int(np.prod(self.input_shape)),)

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, d, h, w)), input values

        :return: np.array((n, (d * h * w))), output values

            n - batch size
            d - number of input channels
            (h, w) - image shape
        """
        # your code here \/
        return inputs.reshape(inputs.shape[0], -1)
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, (d * h * w))), dLoss/dOutputs

        :return: np.array((n, d, h, w)), dLoss/dInputs

            n - batch size
            d - number of units
            (h, w) - input image shape
        """
        # your code here \/
        return grad_outputs.reshape(grad_outputs.shape[0], *self.input_shape)
        # your code here /\


# =============================== 4.1.5 Dropout ==============================
class Dropout(Layer):
    def __init__(self, p, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.p = p
        self.forward_mask = None

    def forward_impl(self, inputs):
        """
        :param inputs: np.array((n, ...)), input values

        :return: np.array((n, ...)), output values

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            self.forward_mask = (np.random.uniform(0, 1, inputs.shape) > self.p).astype(np.float32)
            output = inputs * self.forward_mask
        else:
            output = inputs * (1 - self.p)
        return output
        # your code here /\

    def backward_impl(self, grad_outputs):
        """
        :param grad_outputs: np.array((n, ...)), dLoss/dOutputs

        :return: np.array((n, ...)), dLoss/dInputs

            n - batch size
            ... - arbitrary shape (the same for input and output)
        """
        # your code here \/
        if self.is_training:
            grad_inputs = grad_outputs * self.forward_mask
        else:
            grad_inputs = grad_outputs * (1 - self.p)
        return grad_inputs        
        # your code here /\


# ====================== 2.3 Train and Test on CIFAR-10 ======================
def train_cifar10_model(x_train, y_train, x_valid, y_valid):
    # your code here \/
    # 1) Create a Model
    model = ...
    # 2) Add layers to the model
    #   (don't forget to specify the input shape for the first layer)

    print(model)

    # 3) Train and validate the model using the provided data
    

    # your code here /\
    return model


# ============================================================================
