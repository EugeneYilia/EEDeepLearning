package com.jstarcraft.module.environment;

import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.factory.Nd4j;

import com.jstarcraft.core.utility.NameThreadFactory;

class EnvironmentThreadFactory extends NameThreadFactory {

	private static final AffinityManager manager = Nd4j.getAffinityManager();

	private EnvironmentContext context;

	public EnvironmentThreadFactory(EnvironmentContext context) {
		super(context.getClass().getName());
		this.context = context;
	}

	@Override
	public Thread newThread(Runnable runnable) {
		int index = number.getAndIncrement();
		String name = group.getName() + COLON + index;
		Thread thread = new EnvironmentThread(context, group, runnable, name, 0);
		manager.attachThreadToDevice(thread, index % manager.getNumberOfDevices());
		return thread;
	}

}