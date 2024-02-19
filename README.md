# SnapFlow

![SnapFlow](readme/snapflow_logo.png)

SnapFlow is a cutting-edge library designed to streamline the creation of pipelines for surrogate modeling within the domain of multiphase fluid dynamics simulations. By leveraging data-driven approaches, SnapFlow facilitates the development, validation, and prediction processes in computational fluid dynamics (CFD) projects. Currently, SnapFlow supports linear and nonlinear dimensionality reductions as well as Dynamic Mode Decomposition (DMD) and Neural Networks-based surrogate modeling.

## Structure

SnapFlow is organized into several key components to enhance usability and flexibility:

- `pipelines/`: This directory hosts the core of SnapFlow's functionality, including:
  - `Tutorials and Template/`: Contains step-by-step tutorials and a template to get you started with SnapFlow.
    - `00_template_pipeline`: A basic template for creating your own pipelines.
    - `01_autoencoder_validation`: Tutorial on validating autoencoder models within SnapFlow.
    - `02_parameter_simulation_prediction`: Demonstrates how to use SnapFlow for simulation predictions based on varying parameters.
  - `Custom Pipelines/`: A space for users to develop and store their own custom pipelines tailored to specific projects.
- `snapflow/`: Contains the source code that powers the SnapFlow library, encapsulating the core functionality and algorithms.
- `readme/`: Directory containing information about the library as well as a comprehensive documentation detailing SnapFlow's capabilities, usage, and examples.
- `Makefile`: Simplifies the installation and setup process through make commands.
- `README.md`: Provides an overview and basic documentation of the SnapFlow project.
- `requirements.txt`: Lists all Python dependencies required to run SnapFlow.


## Installation

To install SnapFlow and set up the necessary environment, follow these steps:

1. Clone the SnapFlow repository to your local machine.
2. Navigate to the SnapFlow directory and run the following commands:

```bash
make create_environment
make requirements
make vs_code_install
```

## Getting Started

To dive into SnapFlow, we recommend the following steps:

1. **Explore the Tutorials**: The `pipelines/Tutorials and Template/` directory contains hands-on tutorials that cover the basics of creating and understanding pipelines in SnapFlow.
2. **Read the Documentation**: The `documentation.pdf` provides in-depth information about SnapFlow's architecture, functionalities, and how to leverage its features for your projects.
3. **Experiment with Custom Pipelines**: Use the insights gained from tutorials and documentation to build your own pipelines in the `pipelines/Custom Pipelines/` directory.

## Support

For support, please refer to the documentation or the README.md files within each directory for specific guidance. Should you encounter any issues or have questions, feel free to open an issue in the GitHub repository.

---
