import { app } from "../../../scripts/app.js";

app.registerExtension({
    name: "RealtimeLoraTrainer.DynamicInputs",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        // Apply to both trainer nodes
        if (!["RealtimeLoraTrainer", "SDXLLoraTrainer"].includes(nodeData.name)) {
            return;
        }

        nodeType.prototype.onNodeCreated = function () {
            // Create the button widget
            const button = this.addWidget("button", "Update inputs", null, () => {
                if (!this.inputs) {
                    this.inputs = [];
                }

                const target_number_of_inputs = this.widgets.find(w => w.name === "inputcount")["value"];
                const num_image_inputs = this.inputs.filter(input => input.type === "IMAGE").length;

                if (target_number_of_inputs === num_image_inputs) return;

                if (target_number_of_inputs < num_image_inputs) {
                    // Remove excess inputs (from the end) - both image and caption
                    for (let i = num_image_inputs; i > target_number_of_inputs; i--) {
                        // Find and remove caption_N first, then image_N
                        for (let j = this.inputs.length - 1; j >= 0; j--) {
                            if (this.inputs[j].name === `caption_${i}`) {
                                this.removeInput(j);
                                break;
                            }
                        }
                        for (let j = this.inputs.length - 1; j >= 0; j--) {
                            if (this.inputs[j].name === `image_${i}`) {
                                this.removeInput(j);
                                break;
                            }
                        }
                    }
                } else {
                    // Add new inputs interleaved (image then caption for each)
                    for (let i = num_image_inputs + 1; i <= target_number_of_inputs; ++i) {
                        this.addInput(`image_${i}`, "IMAGE");
                        this.addInput(`caption_${i}`, "STRING");
                    }
                }
            });

            // Move button to right after inputcount widget
            const inputcountIndex = this.widgets.findIndex(w => w.name === "inputcount");
            if (inputcountIndex !== -1) {
                // Remove button from end and insert after inputcount
                this.widgets.pop();
                this.widgets.splice(inputcountIndex + 1, 0, button);
            }
        };
    }
});
