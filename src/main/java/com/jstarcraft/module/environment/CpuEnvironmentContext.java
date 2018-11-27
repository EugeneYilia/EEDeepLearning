package com.jstarcraft.module.environment;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

import org.apache.commons.math3.util.FastMath;

import com.jstarcraft.core.utility.HashUtility;

/**
 * CPU环境上下文
 * 
 * @author Birdy
 *
 */
class CpuEnvironmentContext extends EnvironmentContext {

	static final CpuEnvironmentContext INSTANCE;

	static {
		INSTANCE = new CpuEnvironmentContext();
		INSTANCE.numberOfThreads = Runtime.getRuntime().availableProcessors();
		{
			int numberOfTasks = 1;
			EnvironmentThreadFactory factory = new EnvironmentThreadFactory(INSTANCE);
			INSTANCE.taskExecutor = Executors.newFixedThreadPool(numberOfTasks, factory);
		}
		{
			EnvironmentThreadFactory factory = new EnvironmentThreadFactory(INSTANCE);
			INSTANCE.algorithmExecutors = new ExecutorService[INSTANCE.numberOfThreads];
			for (int threadIndex = 0; threadIndex < INSTANCE.numberOfThreads; threadIndex++) {
				INSTANCE.algorithmExecutors[threadIndex] = Executors.newSingleThreadExecutor(factory);
			}
		}
		{
			EnvironmentThreadFactory factory = new EnvironmentThreadFactory(INSTANCE);
			INSTANCE.structureExecutors = new ExecutorService[INSTANCE.numberOfThreads];
			for (int threadIndex = 0; threadIndex < INSTANCE.numberOfThreads; threadIndex++) {
				INSTANCE.structureExecutors[threadIndex] = Executors.newSingleThreadExecutor(factory);
			}
		}
	}

	private int numberOfThreads;

	private ExecutorService taskExecutor;

	private ExecutorService[] algorithmExecutors;

	private ExecutorService[] structureExecutors;

	private CpuEnvironmentContext() {
	}

	@Override
	public Future<?> doTask(Runnable command) {
		Future<?> task = taskExecutor.submit(() -> {
			int size = 1024 * 1024 * 10;
			{
				EnvironmentThread thread = EnvironmentThread.currentThread();
				thread.createCache(size);
			}
			doAlgorithmByEvery(() -> {
				EnvironmentThread thread = EnvironmentThread.currentThread();
				thread.createCache(size);
			});
			doStructureByEvery(() -> {
				EnvironmentThread thread = EnvironmentThread.currentThread();
				thread.createCache(size);
			});
			command.run();
			doStructureByEvery(() -> {
				EnvironmentThread thread = EnvironmentThread.currentThread();
				thread.deleteCache();
			});
			doAlgorithmByEvery(() -> {
				EnvironmentThread thread = EnvironmentThread.currentThread();
				thread.deleteCache();
			});
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
		int threadIndex = FastMath.abs(HashUtility.twNumberHash32(code)) % numberOfThreads;
		algorithmExecutors[threadIndex].execute(command);
	}

	@Override
	public synchronized void doAlgorithmByEvery(Runnable command) {
		try {
			for (int threadIndex = 0; threadIndex < numberOfThreads; threadIndex++) {
				algorithmExecutors[threadIndex].submit(() -> {
					command.run();
				}).get();
			}
		} catch (Exception exception) {
			throw new RuntimeException(exception);
		}
	}

	@Override
	public void doStructureByAny(int code, Runnable command) {
		int threadIndex = FastMath.abs(HashUtility.twNumberHash32(code)) % numberOfThreads;
		structureExecutors[threadIndex].execute(command);
	}

	@Override
	public synchronized void doStructureByEvery(Runnable command) {
		CountDownLatch latch = new CountDownLatch(numberOfThreads);
		for (int threadIndex = 0; threadIndex < numberOfThreads; threadIndex++) {
			structureExecutors[threadIndex].execute(() -> {
				command.run();
				latch.countDown();
			});
		}
		try {
			latch.await();
		} catch (Exception exception) {
			throw new RuntimeException(exception);
		}
	}

}
