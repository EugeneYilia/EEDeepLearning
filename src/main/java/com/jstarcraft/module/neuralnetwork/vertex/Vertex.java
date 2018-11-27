package com.jstarcraft.module.neuralnetwork.vertex;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.math.structure.matrix.MathMatrix;
import com.jstarcraft.module.neuralnetwork.Model;

/**
 * <pre>
 * A GraphVertex is a vertex in the computation graph. It may contain Layer, or define some arbitrary forward/backward pass
 * behaviour based on the inputs.<br>
 * The purposes of GraphVertex instances are as follows:
 * 1. To track the (local) network connection structure: i.e., a GraphVertex knows about the vertices on the input and output sides
 * 2. To store intermediate results (activations and epsilons)
 * 3. To allow forward pass and backward pass to be conducted, once the intermediate results are set
 * </pre>
 * 
 * @author Alex Black
 */
public interface Vertex extends Model {

	/**
	 * 根据指定的样本分配缓存(每次epoch调用)
	 * 
	 * @param samples
	 */
	void doCache(KeyValue<MathMatrix, MathMatrix>... samples);

	/**
	 * Get the name/label of the GraphVertex
	 */
	String getVertexName();

	/**
	 * A representation of the vertices that are inputs to this vertex (inputs duing
	 * forward pass)<br>
	 * Specifically, if inputVertices[X].getVertexIndex() = Y, and
	 * inputVertices[X].getVertexEdgeNumber() = Z then the Zth output connection
	 * (see {@link #getNumberOfOutputs()} of vertex Y is the Xth input to this
	 * vertex
	 */
	KeyValue<MathMatrix, MathMatrix> getInputKeyValue(int position);

	/**
	 * A representation of the vertices that this vertex is connected to (outputs
	 * duing forward pass) Specifically, if outputVertices[X].getVertexIndex() = Y,
	 * and outputVertices[X].getVertexEdgeNumber() = Z then the Xth output of this
	 * vertex is connected to the Zth input of vertex Y
	 */
	KeyValue<MathMatrix, MathMatrix> getOutputKeyValue();

}
