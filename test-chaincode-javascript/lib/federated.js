'use strict';

const { Contract } = require('fabric-contract-api');

class Federated extends Contract {
    async InitLedger(ctx) {
        const generalModel = {"vesion" : 0, 
                                "model" : [0,0,0,0,0,0]}
        const partialModels = {"version" : 0, "models" : []}
        await ctx.stub.putState("generalModel", Buffer.from(JSON.stringify(generalModel)));
        ctx.stub.setEvent('init', Buffer.from("general model published"));
        await ctx.stub.putState("partialModels", Buffer.from(JSON.stringify(partialModels)));

    }

    async ReadGeneralModel(ctx) {
        const generalModel = await ctx.stub.getState("generalModel")
        if (!generalModel || generalModel.length === 0) {
            throw new Error(`The general model does not exist`);
        }
        return generalModel.toString();
    }

    async PublishPartialModel(ctx, version, partialModel) {
        var generalModel = await ctx.stub.getState("generalModel")
        generalModel = JSON.parse(generalModel)

        if (generalModel.version > version) {
            return "Your model is not updated";
        }
        var partialModels = await ctx.stub.getState("partialModels")
        partialModels = JSON.parse(partialModels)
        
        if (generalModel.version === version) {
            partialModels.models.push(partialModel.split(",").map(Number))
        } else {
            partialModels.version = version
            partialModels.models = [partialModel.split(",").map(Number)]
        }

        await ctx.stub.putState("partialModels", Buffer.from(JSON.stringify(partialModels)));
        ctx.stub.setEvent('general_model_published', Buffer.from("general model published"));

        return "Model accepted"

    }

    async ReadPartialModels(ctx) {
        const partialModels = await ctx.stub.getState("partialModels")
        if (!partialModels || partialModels.length === 0) {
            throw new Error(`There are no partial models`);
        }
        return partialModels.toString();
    }
}

module.exports = Federated