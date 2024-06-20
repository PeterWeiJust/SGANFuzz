---
# SGANFuzz: A Deep Learning-based MQTT Fuzzing method using Generative Adversarial Networks

SGANFuzz is a fuzzing tool that uses Generative Adversarial Networks (GANs) to generate sequences of MQTT messages that can be used to test the robustness of MQTT brokers and clients. The tool is designed to help developers and security researchers identify vulnerabilities and weaknesses in MQTT implementations.

## Features
- Uses GANs to generate realistic MQTT message sequences
- Supports multiple message types and parameters
- Allows users to configure message sequence length and batch size
- Provides visualization tools to help users analyze generated message sequences

## Dependencies
- Python >= 3.6
- TensorFlow >= 2.9
- Eclipse Mosquitto (for testing)
- EMQX (for testing)
- Windows/Linux OS environment 

## Getting Started
To use SGANFuzz, first install the required dependencies and clone the repository. Next, configure the tool by setting the desired message sequence length, batch size, and other parameters. 
Running the following command for MQTT SeqGAN training:

python train_seqgan.py

Finally, run the tool to generate MQTT message sequences, which can be used to test MQTT brokers and clients:

python send_payload.py

For more detailed instructions and examples, see the [User Guide](user_guide.md) and [Examples](examples/) directory.

## Contributing
Contributions to SGANFuzz are welcome! To contribute, please fork the repository, make your changes, and submit a pull request. For more information, see the [Contributing Guidelines](CONTRIBUTING.md).

## License
SGANFuzz is licensed under the [MIT License](LICENSE).
