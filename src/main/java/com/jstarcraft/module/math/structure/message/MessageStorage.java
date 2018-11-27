package com.jstarcraft.module.math.structure.message;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.concurrent.Semaphore;

import com.jstarcraft.core.utility.KeyValue;
import com.jstarcraft.module.environment.EnvironmentContext;
import com.jstarcraft.module.math.structure.MathMessage;

public class MessageStorage {

	private List<KeyValue<Boolean, MathMessage>> globalKeyValues;

	private ThreadLocal<KeyValue<Boolean, MathMessage>> localKeyValues;

	/** 信号量 */
	private Semaphore semaphore;

	public MessageStorage() {
		globalKeyValues = new LinkedList<>();
		localKeyValues = new ThreadLocal<>();
		semaphore = new Semaphore(0);

		EnvironmentContext context = EnvironmentContext.getContext();
		context.doStructureByEvery(() -> {
			KeyValue<Boolean, MathMessage> keyValue = new KeyValue<>(false, null);
			localKeyValues.set(keyValue);
			synchronized (globalKeyValues) {
				globalKeyValues.add(keyValue);
			}
		});
		globalKeyValues = new ArrayList<>(globalKeyValues);
	}

	public <T extends MathMessage> T attachMessage(T message) {
		if (message == null) {
			return null;
		} else {
			for (KeyValue<Boolean, MathMessage> keyValue : globalKeyValues) {
				if (keyValue.getKey()) {
					message.attach(keyValue.getValue());
					keyValue.setKey(false);
				}
			}
			return message;
		}
	}

	public <T extends MathMessage> T detachMessage(T message) {
		if (message == null) {
			return null;
		} else {
			KeyValue<Boolean, MathMessage> keyValue = localKeyValues.get();
			MathMessage copy;
			if (keyValue.getKey()) {
				copy = keyValue.getValue();
			} else {
				copy = message.detach();
				keyValue.setKey(true);
				keyValue.setValue(copy);
			}
			return (T) copy;
		}
	}

	public Semaphore getSemaphore() {
		return semaphore;
	}

}
