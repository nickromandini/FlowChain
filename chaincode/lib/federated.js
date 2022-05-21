'use strict';

const { Contract } = require('fabric-contract-api');
const pako = require('pako')
const nj = require('numjs')

const bcrypt = require('bcrypt')
const NodeRSA = require('node-rsa');


function zip(arrays) {
    return arrays[0].map(function(_,i){
        return arrays.map(function(array){return array[i]})
    });
}

function mean(arrays) {
    var temp = arrays[0]
    for (var i = 1; i < arrays.length; i++)
        temp = nj.add(temp, arrays[i])
    return temp.divide(2).tolist()
}

function generate_quantization_constants(alpha, beta, alpha_q, beta_q) {
    var s = (beta - alpha) / (beta_q - alpha_q)
    var z = parseInt((beta * alpha_q - alpha * beta_q) / (beta - alpha))
    return [s, z]
}


function generate_quantization_int8_constants(alpha, beta) {
    var b = 8
    var alpha_q = -(2**(b - 1))
    var beta_q = 2**(b - 1) - 1

    return generate_quantization_constants(alpha, beta, alpha_q, beta_q)
}

function find_min_max(weights) {
    var min = 0
    var max = 0
    for(var l of weights){
        var min_t = nj.min(l)
        var max_t = nj.max(l)
        if (min_t < min)
            min = min_t
        if (max_t > max)
            max = max_t
    }
    return [min, max]
}


function quantization(x, s, z, alpha_q, beta_q) {
    var x_q = nj.round(nj.add(nj.multiply(x, 1/s),z))
    x_q = nj.clip(x_q, alpha_q, beta_q)

    return x_q.tolist()
}


function quantization_int8(x, s, z) {
    var x_q = quantization(x, s, z, -128, 127)
    //x_q = x_q.astype(np.int8)

    return x_q
}

function quantization_layers_int8(model) {
    var model_q = []
    var parameters = []
    for(var l of model){
        var min = nj.min(l)
        var max = nj.max(l)
        var s_z = generate_quantization_int8_constants(min, max)
        model_q.push(quantization_int8(l, s_z[0], s_z[1]))
        parameters.push(s_z)
    }
    return [parameters, model_q]
}


function dequantization(x_q, s, z) {
    var x = nj.subtract(x_q,z).multiply(s)
    //x = x.astype(np.float32)

    return x.tolist()
}

function dequantization_layers(parameters, model) {
    var res = []
    for(var i in model)
        res.push(dequantization(model[i], parameters[i][0], parameters[i][1]))
    return res
}




class Federated extends Contract {

    


    async Init(ctx) {
        //const generalModel = [[0,0],[[0,0,0]]]
        const partialModelsIDs = []
        //await ctx.stub.putState("generalModel", pako.deflate(JSON.stringify(generalModel)));
        await ctx.stub.putState("generalModelVersion", JSON.stringify(0))
        //ctx.stub.setEvent('init', Buffer.from("general model published"));
        await ctx.stub.putState("partialModelsIDs", JSON.stringify(partialModelsIDs));
        return "Ledger initialized"

    }

    async ReadGeneralModel(ctx) {
        var generalModel = await ctx.stub.getState("generalModel")
        if (!generalModel) {
            throw new Error(`The general model does not exist`);
        }
        generalModel = JSON.parse(pako.inflate(generalModel, { to: 'string' }))
        var generalModelVersion = await ctx.stub.getState("generalModelVersion")
        generalModelVersion = JSON.parse(generalModelVersion)
        console.info(generalModel[0])
        return JSON.stringify([generalModelVersion, generalModel])
    }

    async PublishPartialModel(ctx, partialModel) {
        console.info("here")
        //var partialModel = JSON.parse(partialModel)
        //var partialModelVersion = partialModel[0]
        //partialModel = partialModel[1]
        var generalModelVersion = await ctx.stub.getState("generalModelVersion")
        generalModelVersion = JSON.parse(generalModelVersion)
        partialModel = JSON.parse(partialModel)
        //var generalModel = await ctx.stub.getState("generalModel")
        //generalModel = JSON.parse(generalModel)
        //var generalModelVersion = generalModel[0]
        //generalModel = generalModel[1]
        console.info('here1')
        
        //generalModel = JSON.parse(pako.inflate(generalModel, { to: 'string' }))

        if (generalModelVersion !== partialModel.version) {
            return "Error";
        } else {
            var clientID = ctx.clientIdentity.getID()
            
            await ctx.stub.putState("partialModel" + clientID, pako.deflate(JSON.stringify([partialModel.parameters, partialModel.model])))
            var partialModelsIDs = await ctx.stub.getState("partialModelsIDs")
            partialModelsIDs = JSON.parse(partialModelsIDs)
            console.info('here2')
            //partialModels = JSON.parse(pako.inflate(partialModels, { to: 'string' }))

            //partialModels[ctx.clientIdentity.getID()] = partialModel

            //partialModelsIDs.push(clientID)
            
            if(partialModelsIDs.length + 1 === 2) {
                console.info('here3')
                var models = [dequantization_layers(partialModel.parameters, partialModel.model)]
                for(var id of partialModelsIDs) {
                    var parameters_model = await ctx.stub.getState("partialModel"+id)
                    parameters_model = JSON.parse(pako.inflate(parameters_model, {to:"string"}))
                    //temp = [...temp]
                    models.push(dequantization_layers(parameters_model[0], parameters_model[1]))
                }
                /*for(var i in partialModels) {
                    models.push(partialModels[i])
                }*/
                var zippedModels = zip(models)
                var newGM = []
                for(var layer of zippedModels) {
                    newGM.push(mean(layer))
                }
                var parameters_newGM = quantization_layers_int8(newGM)
                console.info('here4')
                await ctx.stub.putState("generalModelVersion", JSON.stringify(generalModelVersion + 1))
                await ctx.stub.putState("generalModel", pako.deflate(JSON.stringify(parameters_newGM)));
                console.info('here5')
                //partialModels = {}
                await ctx.stub.putState("partialModelsIDs", JSON.stringify([]));
                console.info('here6')

                //ctx.stub.setEvent('general_model_published', Buffer.from(JSON.stringify(newGeneralModel)));
                ctx.stub.setEvent('general_model_published')
                console.info('here7')

                return 'Model published and new general model published'

            } else {
                //partialModelsIDs = [...partialModelsIDs]
                partialModelsIDs.push(clientID)
                
                await ctx.stub.putState("partialModelsIDs", JSON.stringify(partialModelsIDs));

                return 'Model published'
            }
        
        }

    }

    async PubPart(ctx, pm) {
        await ctx.stub.putState('test', Buffer.from(pm))
        await ctx.stub.putState('test1', Buffer.from(pm))
        await ctx.stub.putState('test2', Buffer.from(pm))
        await ctx.stub.putState('test3', Buffer.from(pm))

        return "hurray"
    }

    async ReadPartialModels(ctx) {
        var partialModels = await ctx.stub.getState("partialModels")
        partialModels = JSON.parse(partialModels)
        if (!partialModels) {
            throw new Error(`There are no partial models`);
        }
        return JSON.stringify(partialModels)
    }



    async Aggregate(ctx, partialModels) {



        ctx.stub.setEvent('general_model_published', Buffer.from("general model published"));
    }
}


c
module.exports.Federated = Federated
