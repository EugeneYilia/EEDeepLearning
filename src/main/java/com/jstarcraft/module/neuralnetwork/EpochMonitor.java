package com.jstarcraft.module.neuralnetwork;

public interface EpochMonitor {

	void beforeForward();

	void afterForward();

	void beforeBackward();

	void afterBackward();

}
