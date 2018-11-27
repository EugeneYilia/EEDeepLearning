package com.jstarcraft.module.neuralnetwork.layer;

import com.jstarcraft.core.utility.RandomUtility;
import com.jstarcraft.module.math.structure.MathCalculator;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.schedule.Schedule;

/**
 * Implements standard (inverted) dropout.<br>
 * <br>
 * Regarding dropout probability. This is the probability of <it>retaining</it>
 * each input activation value for a layer. Thus, each input activation x is
 * independently set to:<br>
 * x <- 0, with probability 1-p<br>
 * x <- x/p with probability p<br>
 * Note that this "inverted" dropout scheme maintains the expected value of
 * activations - i.e., E(x) is the same before and after dropout.<br>
 * Dropout schedules (i.e., varying probability p as a function of
 * iteration/epoch) are also supported.<br>
 * <br>
 * Other libraries (notably, Keras) use p == probability(<i>dropping</i> an
 * activation)<br>
 * In DL4J, {@code new Dropout(x)} will keep an input activation with
 * probability x, and set to 0 with probability 1-x.<br>
 * Thus, a dropout value of 1.0 is functionally equivalent to no dropout: i.e.,
 * 100% probability of retaining each input activation.<br>
 * <p>
 * Note 1: As per all IDropout instances, dropout is applied at training time
 * only - and is automatically not applied at test time (for evaluation,
 * etc)<br>
 * Note 2: Care should be taken when setting lower (probability of retaining)
 * values for (too much information may be lost with aggressive (very low)
 * dropout values).<br>
 * Note 3: Frequently, dropout is not applied to (or, has higher retain
 * probability for) input (first layer) layers. Dropout is also often not
 * applied to output layers.<br>
 * Note 4: Implementation detail (most users can ignore): DL4J uses inverted
 * dropout, as described here: <a href=
 * "http://cs231n.github.io/neural-networks-2/">http://cs231n.github.io/neural-networks-2/</a>
 * </p>
 * <br>
 * See: Srivastava et al. 2014: Dropout: A Simple Way to Prevent Neural Networks
 * from Overfitting <a href=
 * "http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf">http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf</a>
 *
 * @author Alex Black
 */
public class DefaultMasker implements Masker {

	private Schedule schedule;

	public DefaultMasker(Schedule schedule) {
		this.schedule = schedule;
	}

	@Override
	public void mask(MathMatrix matrix, int iteration, int epoch) {
		float current = schedule.valueAt(iteration, epoch);

		matrix.mapValues((row, column, value, message) -> {
			return RandomUtility.randomFloat(1F) < current ? 0F : value;
		}, null, MathCalculator.PARALLEL);
	}

}
