import jax
import jax.numpy as jnp
from flax import nnx

"""
 I HAD DINNER WITH ERIC ENG AND HE PROPOSED TANH AS MY ACTIVATION FUNCTION
 SO I SWITCHED TO THE XAVIER GLOROT KERNEL AND SWITHCED TO TANH AND IT WORKED PROFESSOR
"""

def xavier_kernel_init(num_inputs: int, num_outputs: int):
    """Glorot/Xavier uniform initialization"""
    def init(rng, shape, dtype=jnp.float32):
        limit = jnp.sqrt(6.0 / (num_inputs + num_outputs))
        return jax.random.uniform(rng, shape, dtype, minval=-limit, maxval=limit)
    return init

class Block(nnx.Module):
    """define each hidden layer block"""
    def __init__(self, num_inputs: int, hl_width: int, rngs: nnx.Rngs):
        super().__init__()
        self.linear = nnx.Linear(
            num_inputs,
            hl_width,
            rngs=rngs,
            kernel_init=xavier_kernel_init(num_inputs, hl_width),
        )

    def __call__(self, x: jax.Array):
        x = self.linear(x)        
        """ 
        performs the "mat-mul" as you would say. ☝️🤓☝️🤓☝️🤓 
        idk if just pdf will print the emojis but in case they dont, i pasted the nerd emoji with the 
        finger pointing up because you embody the "ermmm actually" *pushes glasses up* persona.
      """
        return jax.nn.tanh(x)

class MLP(nnx.Module):
    """build our hidden layers"""
    def __init__(
        self,
        *,
        rngs: nnx.Rngs,
        num_inputs: int,
        num_outputs: int,
        num_hl: int,
        hl_width: int,
    ):
        super().__init__() #initialize nnx.Module internal parameters
        self.num_inputs = num_inputs
        self.num_outputs = num_outputs
        self.num_hl = num_hl
        self.hl_width = hl_width

        key = rngs.params()

        #initial layer
        k_first, key = jax.random.split(key)
        self.first = nnx.Linear(
            num_inputs,
            hl_width,
            rngs=nnx.Rngs(k_first),
            kernel_init=xavier_kernel_init(num_inputs, hl_width),
        )

        #hidden layers
        @nnx.split_rngs(splits=num_hl - 1)
        @nnx.vmap(in_axes=(0,), out_axes=0)
        def make_block(rngs_for_one_layer: nnx.Rngs):
            return Block(num_inputs=hl_width, hl_width=hl_width, rngs=rngs_for_one_layer)

        self.hidden_blocks = make_block(nnx.Rngs(key))

        #output layer
        k_out = jax.random.split(key, 2)[1]
        self.out = nnx.Linear(
            hl_width,
            num_outputs,
            rngs=nnx.Rngs(k_out),
            kernel_init=xavier_kernel_init(hl_width, num_outputs=1),
        )

    def __call__(self, x: jax.Array):
        # First transform
        x = jax.nn.tanh(self.first(x))

        #applies Block sequentially to our layers
        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=nnx.Carry)
        def forward(carry, block):
            return block(carry)

        x = forward(x, self.hidden_blocks) 

        #use logits as an input to the sigmoid function in our last matmul
        logits = self.out(x)  
        probs = jax.nn.sigmoid(logits)
        return probs

"""
my first attempt involved using ReLUs as my activation unit with only 2 hidden layers since He talks about how 2 is almost 
always enough. However, i soon realized that i needed greater expressive power because i would get very jagged decision boundaries.
I then increased my num_hl and hl_width to be able to model the curves of the spirals. However, when my num_hl or width grew too much,
my plot would be either all red or all blue which led me to beleive that the more hidden layers, the more vulnerable i am to gradient
explosion or implosion. I found a happy medium with what i currently have in my config file. 

I had dinner with Eric Eng and Isaac Rivera today and they suggested using tanh activation function which they claimed were good for 
circular things. I also switched my kernal initiazliation from the kaiming he to the glorot xavier one because the kaiming one 
was specific to ReLUs. They also told me to just crank up my num_hl and hl_width to increase my expressive power but all that did 
was make my computer turn into a jet turbine. 

In the end, im happy with tanh as my activation functin and my parameters in the config file.
"""