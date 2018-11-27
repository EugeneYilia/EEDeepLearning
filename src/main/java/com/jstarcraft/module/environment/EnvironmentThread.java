package com.jstarcraft.module.environment;

import org.nd4j.linalg.api.memory.MemoryWorkspace;
import org.nd4j.linalg.api.memory.conf.WorkspaceConfiguration;
import org.nd4j.linalg.api.memory.enums.AllocationPolicy;
import org.nd4j.linalg.api.memory.enums.LearningPolicy;
import org.nd4j.linalg.factory.Nd4j;

/**
 * 环境线程
 * 
 * @author Birdy
 *
 */
public class EnvironmentThread extends Thread {

	private EnvironmentContext context;

	private float[] array;

	private MemoryWorkspace space;

	EnvironmentThread(EnvironmentContext context, ThreadGroup group, Runnable runnable, String name, long size) {
		super(group, runnable, name, size);
		this.context = context;
	}

	public EnvironmentContext getContext() {
		return context;
	}

	void createCache(int size) {
		array = new float[size];
		WorkspaceConfiguration configuration = WorkspaceConfiguration.builder().initialSize(size).policyAllocation(AllocationPolicy.STRICT).policyLearning(LearningPolicy.NONE).build();
		space = Nd4j.getWorkspaceManager().createNewWorkspace(configuration, "ND4J");
	}

	void deleteCache() {
		array = null;
		Nd4j.getWorkspaceManager().destroyWorkspace(space);
		space = null;
	}

	public float[] getArray() {
		return array;
	}

	public MemoryWorkspace getSpace() {
		return space.notifyScopeEntered();
	}

	public static EnvironmentThread currentThread() {
		return EnvironmentThread.class.cast(Thread.currentThread());
	}

}
