#pragma once

#include <stack>
#include <string>

struct TestMethodCollector
{
	static std::stack<TestMethodCollector*> instances;

	TestMethodCollector() { TestMethodCollector::instances.push(this); }

	virtual std::string getName() const = 0;
	virtual void invoke() = 0;
};

#define TEST_METHOD(testName) \
struct testName : public TestMethodCollector \
{ \
	static testName si; \
	std::string getName() const override { return #testName; } \
	void invoke() override; \
}; \
testName testName::si; \
void testName::invoke() \
/* */
