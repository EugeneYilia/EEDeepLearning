package com.jstarcraft.module.neuralnetwork;

import org.junit.Test;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.factory.Nd4j;

public class Nd4jTestCase {

	@Test
	public void testAffinityManager() {
		AffinityManager manager = Nd4j.getAffinityManager();
		System.out.println(manager.getDeviceForCurrentThread());
		System.out.println(manager.getNumberOfDevices());
		System.out.println(manager.getClass().getSimpleName());
	}

}
