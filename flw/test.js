var nj = require('numjs');

var fs = require('fs');

//zip= rows=>rows[0].map((_,c)=>rows.map(row=>row[c]))

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

try {
    var data = fs.readFileSync('sample.txt', 'utf8');
    //console.log(data);    
} catch(e) {
    console.log('Error:', e.stack);
}

var model = JSON.parse(data)


var model2 = [[1],[2],[3]]

var models = []
models.push(model.model)
models.push(model.model)
models.push(model.model)
models.push(model.model)
var weights_prime = []

//var k = nj.add(...models).tolist()


var zipped = zip(models)

for(var l of zipped) {
    //var temp = zipped[l]
    //var i = nj.add(...temp).tolist()
    //var j = sumArrays(zipped[l]).tolist()
    weights_prime.push(mean(l))
    //var h = j.tolist()
    var o = 1
}
console.log(weights_prime)