package com.jstarcraft.module.environment;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.nd4j.linalg.factory.Nd4j;

/**
 * GPU环境上下文
 * 
 * @author Birdy
 *
 */
class GpuEnvironmentContext extends EnvironmentContext {

	static final GpuEnvironmentContext INSTANCE;

	static {
		INSTANCE = new GpuEnvironmentContext();
		int numberOfThreads = Nd4j.getAffinityManager().getNumberOfDevices();
		EnvironmentThreadFactory factory = new EnvironmentThreadFactory(INSTANCE);
		INSTANCE.executor = Executors.newFixedThreadPool(numberOfThreads, factory);
	}

	private ExecutorService executor;

	private GpuEnvironmentContext() {
	}

	@Override
	public Future<?> doTask(Runnable command) {
		Future<?> task = executor.submit(() -> {
			int size = 1024 * 1024 * 10;
			{
				EnvironmentThread thread = EnvironmentThread.currentThread();
				thread.createCache(size);
			}
			command.run();
			{
				EnvironmentThread thread = EnvironmentThread.currentThread();
				thread.deleteCache();
			}
			// 必须触发垃圾回收.
			System.gc();
		});
		return task;
	}

	@Override
	public void doAlgorithmByAny(int code, Runnable command) {
		command.run();
	}

	@Override
	public void doAlgorithmByEvery(Runnable command) {
		command.run();
	}

	@Override
	public void doStructureByAny(int code, Runnable command) {
		command.run();
	}

	@Override
	public void doStructureByEvery(Runnable command) {
		command.run();
	}

}
