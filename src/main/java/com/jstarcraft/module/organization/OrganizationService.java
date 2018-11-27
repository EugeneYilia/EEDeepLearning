package com.jstarcraft.module.organization;

public interface OrganizationService {

	void check(User user, Group group, String resourceConfiguration);

	<T extends Resource> T getResource(Class<T> clazz, User user, Group group, Long resourceId);
	
	<T extends Resource> T createResource(Class<T> clazz, User user, Group group);
	
	<T extends Resource> T deleteResource(Class<T> clazz, User user, Group group, Long resourceId);

}
