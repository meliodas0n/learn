
function Header(props) {
  const title = props.title
  return <div>{title ? title : 'Default Title'}</div>
}

function HomePage() {
  const names = ['Ada Lovelace', 'Grace Hopper', 'Margaret Hamilton']

  return (
    <div>
      <Header title="Develop. Preview. Ship." />
      <ul>
        {names.map(name => (
          <li>{name}</li>
        ))}
      </ul>
    </div>
  )
}