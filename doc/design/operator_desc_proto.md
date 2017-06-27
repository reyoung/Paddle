# Protobuf Message for Operator


The Operator factory method, `CreateOperator` should take a `protobuf` message, because we currently use protobuf as neural network's serialization protocol. The `CreateOperator` is a global function, and the interface should be

```cpp
extern OperatorBase* CreateOperator(const OperatorDesc& desc);
```

The `OperatorDesc` should be the protobuf message for all operator. The protobuf message should represent all private member for each operator.

The `OperatorDesc` should contain following fields:

1. It stores all input and output variable names because `Scope` holds all variables' information.
1. It stores the type name of the current operator.
1. It stores the attribute map, which is an association from attribute name to attribute value. The design of Attribute is written by [Attribute](./attribute.md) design doc.

The `OperatorDesc` should be

```proto
message OperatorDesc {
    repeated string inputs = 1;
    repeated string outputs = 2;
    required string type = 3;
    map<string, Attribute> attrs = 4;
}
```
