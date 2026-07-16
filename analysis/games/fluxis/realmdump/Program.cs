// Dump a fluXis realm database to JSON using Realm's dynamic-schema API.
//
// Invariants the caller (analysis/games/fluxis/realm_reader.py) upholds:
//   * runs against a COPY of fluxis.realm -- opening with a newer Realm
//     SDK upgrades the file format in place, and the game must never
//     see that;
//   * the path argument is ABSOLUTE -- Realm resolves relative paths
//     against its own default data folder and silently creates a fresh
//     empty database there instead of opening the copy.
using System.Text.Json;
using Realms;

var path = args[0];
var config = new RealmConfiguration(path) { IsDynamic = true };
using var realm = Realm.GetInstance(config);

var output = new Dictionary<string, object>();
var schemaInfo = new Dictionary<string, List<string>>();

foreach (var objectSchema in realm.Schema)
{
    var props = new List<string>();
    foreach (var p in objectSchema)
        props.Add($"{p.Name}:{p.Type}");
    schemaInfo[objectSchema.Name] = props;
}
output["schema"] = schemaInfo;

foreach (var tableName in new[]
         { "RealmScore", "RealmMap", "RealmMapSet", "RealmMapMetadata" })
{
    if (!realm.Schema.TryFindObjectSchema(tableName, out var objectSchema))
        continue;
    var rows = new List<Dictionary<string, object?>>();
    foreach (var obj in realm.DynamicApi.All(tableName))
    {
        var row = new Dictionary<string, object?>();
        foreach (var prop in objectSchema)
        {
            if (prop.Type.HasFlag(Realms.Schema.PropertyType.Array))
                continue;
            if (prop.Type.HasFlag(Realms.Schema.PropertyType.Object))
            {
                // Links to keyed objects flatten to "<Prop>ID"; embedded
                // objects (no ID property, e.g. RealmMapMetadata) inline
                // their scalars as "<Prop>.<SubProp>" so the Python side
                // never needs nested traversal.
                try
                {
                    var linked = obj.DynamicApi.Get<IRealmObject?>(prop.Name);
                    if (linked is null)
                    {
                        row[prop.Name + "ID"] = null;
                        continue;
                    }
                    if (linked.ObjectSchema!.TryFindProperty("ID", out _))
                    {
                        row[prop.Name + "ID"] =
                            linked.DynamicApi.Get<object?>("ID")?.ToString();
                        continue;
                    }
                    foreach (var sub in linked.ObjectSchema!)
                    {
                        if (sub.Type.HasFlag(Realms.Schema.PropertyType.Object) ||
                            sub.Type.HasFlag(Realms.Schema.PropertyType.Array))
                            continue;
                        var sv = linked.DynamicApi.Get<object?>(sub.Name);
                        row[$"{prop.Name}.{sub.Name}"] = sv switch
                        {
                            Guid g => g.ToString(),
                            DateTimeOffset d => d.ToString("o"),
                            _ => sv,
                        };
                    }
                }
                catch
                {
                    row[prop.Name + "ID"] = null;
                }
                continue;
            }
            try
            {
                var value = obj.DynamicApi.Get<object?>(prop.Name);
                row[prop.Name] = value switch
                {
                    Guid g => g.ToString(),
                    DateTimeOffset d => d.ToString("o"),
                    _ => value,
                };
            }
            catch
            {
                row[prop.Name] = null;
            }
        }
        rows.Add(row);
    }
    output[tableName] = rows;
}

Console.WriteLine(JsonSerializer.Serialize(output));
